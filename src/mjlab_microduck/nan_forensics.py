"""Extra evidence for mjlab's NaN guard: what drove the offending physics step.

mjlab's guard keeps only qpos/qvel/act before each physics step and saves no
post-step state. That was not enough to find the 2026-10-24 course NaN. This
module extends the guard, when it is enabled, to also keep:

* per physics step, before stepping: ctrl, qacc_warmstart (solver warm start),
  qacc, actuator_force, qfrc_actuator, qfrc_constraint, qfrc_applied
  and sim time, for every environment (small per-env vectors);
* per control step: the raw policy action and each action term's processed
  action;
* the contacts and constraint forces of the last finite step;
* on detection, the offending post-step state: positions, velocities,
  accelerations, forces, sensors, solver iterations, contact/constraint rows;
* every per-environment randomized model field (``sim.expanded_fields``) and any
  per-environment tensor kept on scene entities (e.g. BAM ``friction_scale``).

The evidence file ``nan_forensics_<timestamp>.npz`` is written **before** the
guard prints its detection message, because the session watcher kills training
as soon as that message appears. Disabled guards add no work.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_INSTALLED = False
STEP_FIELDS = ('ctrl', 'qacc_warmstart', 'qacc', 'actuator_force', 'qfrc_actuator',
               'qfrc_constraint', 'qfrc_applied', 'time')
POST_FIELDS = ('qpos', 'qvel', 'act', 'qacc', 'qacc_warmstart', 'ctrl', 'actuator_force',
               'qfrc_actuator', 'qfrc_constraint', 'qfrc_applied', 'qfrc_bias', 'qfrc_passive',
               'sensordata', 'solver_niter', 'nefc', 'time', 'xpos', 'xquat')
CONTACT_FIELDS = ('dist', 'pos', 'frame', 'friction', 'solref', 'solimp', 'dim', 'geom',
                  'efc_address', 'includemargin')
EFC_FIELDS = ('type', 'id', 'pos', 'vel', 'aref', 'force', 'frictionloss', 'margin', 'state')


def _np(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _field(struct, name):
    try:
        value = getattr(struct, name)
    except AttributeError:
        return None
    value = getattr(value, '_tensor', value)  # mjlab TorchArray wraps a torch view.
    return value if isinstance(value, torch.Tensor) else None


def contacts_for(data, envs) -> dict[str, np.ndarray]:
    """Contacts of selected worlds from the global contact arrays."""
    out = {}
    contact = data.contact
    nacon = _field(data, 'nacon')
    count = int(nacon.reshape(-1)[0]) if nacon is not None else 0
    world = _field(contact, 'worldid')
    if world is None or count == 0:
        return out
    count = min(count, world.shape[0])
    world = world[:count]
    for env in envs:
        index = torch.nonzero(world == env).flatten()
        out[f'env{env}/contact_index'] = _np(index)
        for name in CONTACT_FIELDS:
            value = _field(contact, name)
            if value is not None:
                out[f'env{env}/contact_{name}'] = _np(value[:count][index])
    return out


def efc_for(data, envs) -> dict[str, np.ndarray]:
    """Constraint rows (forces etc.) of selected worlds, trimmed to nefc."""
    out = {}
    nefc = _field(data, 'nefc')
    for env in envs:
        n = int(_np(nefc[env])) if nefc is not None else None
        for name in EFC_FIELDS:
            value = _field(data.efc, name)
            if value is not None and value.ndim >= 2 and value.shape[0] > env:
                row = value[env]
                out[f'env{env}/efc_{name}'] = _np(row[:n] if n is not None else row)
    return out


def entity_tensors(env, num_envs: int, envs, max_depth: int = 3) -> dict[str, np.ndarray]:
    """Per-environment tensors held by scene entities and their actuators."""
    out, seen = {}, set()
    def visit(obj, prefix, depth):
        if id(obj) in seen or depth > max_depth:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if obj.ndim >= 1 and obj.shape[0] == num_envs and obj.numel() <= num_envs * 4096:
                for e in envs:
                    out[f'env{e}/{prefix}'] = _np(obj[e])
            return
        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                visit(item, f'{prefix}[{i}]', depth + 1)
        elif isinstance(obj, dict):
            for key, item in obj.items():
                visit(item, f'{prefix}.{key}', depth + 1)
        elif hasattr(obj, '__dict__') and type(obj).__module__.split('.')[0] in ('mjlab', 'mjlab_microduck'):
            for key, item in vars(obj).items():
                visit(item, f'{prefix}.{key}', depth + 1)
    try:
        for name, entity in env.scene.entities.items():
            visit(entity, f'entity.{name}', 0)
    except Exception as error:  # Forensics must never mask the original NaN report.
        out['entity_scan_error'] = np.array(repr(error))
    return out


def install() -> None:
    """Patch mjlab's guard and env step. Idempotent and safe if disabled."""
    global _INSTALLED
    if _INSTALLED:
        return
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.sim import sim as sim_module
    from mjlab.utils.nan_guard import NanGuard

    original_init = sim_module.Simulation.__init__
    original_capture = NanGuard.capture
    original_dump = NanGuard._dump_buffer
    original_env_step = ManagerBasedRlEnv.step

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        guard = self.nan_guard
        guard.sim = self
        if guard.enabled:
            guard.extra = deque(maxlen=guard.buffer_size)
            guard.actions = deque(maxlen=guard.buffer_size)
            guard.last_contacts = None

    def capture(self, wp_data):
        if self.enabled:
            record = {'step': self.step_counter}
            for name in STEP_FIELDS:
                value = _field(wp_data, name)
                if value is not None:
                    record[name] = value.clone()
            self.extra.append(record)
            # Keep only the latest contact/constraint snapshot: these arrays
            # are large. They describe the last finite step's contacts.
            self.last_contacts = {'step': self.step_counter - 1, 'data': {
                f'contact_{n}': _field(wp_data.contact, n).clone() for n in CONTACT_FIELDS + ('worldid',)
                if _field(wp_data.contact, n) is not None}}
            for n in EFC_FIELDS:
                value = _field(wp_data.efc, n)
                if value is not None:
                    self.last_contacts['data'][f'efc_{n}'] = value.clone()
            for n in ('nacon', 'nefc'):
                value = _field(wp_data, n)
                if value is not None:
                    self.last_contacts['data'][n] = value.clone()
        original_capture(self, wp_data)

    def dump(self, nan_env_ids):
        try:
            write_forensics(self, nan_env_ids)
        except Exception as error:
            print(f'[NanForensics] Failed to write forensics: {error!r}', flush=True)
        original_dump(self, nan_env_ids)

    def env_step(self, action):
        guard = getattr(self.sim, 'nan_guard', None)
        if guard is not None and guard.enabled:
            guard.env = self
            record = {'env_step': int(self.common_step_counter), 'physics_step': guard.step_counter,
                      'action': action.detach().clone()}
            guard.actions.append(record)
            result = original_env_step(self, action)
            try:
                for name, term in self.action_manager._terms.items():
                    processed = getattr(term, 'processed_actions', None)
                    if isinstance(processed, torch.Tensor):
                        record[f'processed/{name}'] = processed.detach().clone()
            except Exception:
                pass
            return result
        return original_env_step(self, action)

    sim_module.Simulation.__init__ = init
    NanGuard.capture = capture
    NanGuard._dump_buffer = dump
    ManagerBasedRlEnv.step = env_step
    _INSTALLED = True


def write_forensics(guard, nan_env_ids) -> Path:
    envs = list(nan_env_ids[: guard.max_envs_to_dump])
    sim = guard.sim
    data = sim.data
    out: dict[str, np.ndarray] = {}
    for item in guard.extra:
        for name, value in item.items():
            if name != 'step':
                out[f'pre/{item["step"]:06d}/{name}'] = _np(value[envs])
    for item in guard.actions:
        for name, value in item.items():
            if isinstance(value, torch.Tensor):
                out[f'action/{item["env_step"]:06d}/{name}'] = _np(value[envs])
        out[f'action/{item["env_step"]:06d}/physics_step'] = np.array(item['physics_step'])
    for name in POST_FIELDS:
        value = _field(data, name)
        if value is not None:
            out[f'post/{name}'] = _np(value[envs])
    for key, value in contacts_for(data, envs).items():
        out[f'post/{key}'] = value
    for key, value in efc_for(data, envs).items():
        out[f'post/{key}'] = value
    if guard.last_contacts is not None:
        snapshot = guard.last_contacts['data']
        class View:  # Present the snapshot with the same access pattern as data.
            pass
        view, contact, efc = View(), View(), View()
        view.contact, view.efc = contact, efc
        for key, value in snapshot.items():
            if key.startswith('contact_'):
                setattr(contact, key[8:], value)
            elif key.startswith('efc_'):
                setattr(efc, key[4:], value)
            else:
                setattr(view, key, value)
        for key, value in contacts_for(view, envs).items():
            out[f'last_finite/{key}'] = value
        for key, value in efc_for(view, envs).items():
            out[f'last_finite/{key}'] = value
        out['last_finite/step'] = np.array(guard.last_contacts['step'])
    for name in sorted(getattr(sim, 'expanded_fields', ()) or ()):
        value = _field(sim.model, name)
        if value is not None and value.shape[0] == guard.num_envs:
            out[f'model/{name}'] = _np(value[envs])
    env = getattr(guard, 'env', None)
    if env is not None:
        out.update(entity_tensors(env, guard.num_envs, envs))
        out['episode_length_buf'] = _np(env.episode_length_buf[envs])
        out['env_origins'] = _np(env.scene.env_origins[envs])
        out['common_step_counter'] = np.array(int(env.common_step_counter))
    out['_metadata'] = np.array({
        'nan_env_ids': list(nan_env_ids), 'dumped_env_ids': envs,
        'detection_step': guard.step_counter, 'physics_dt': float(guard.mj_model.opt.timestep),
        'note': 'pre/<step>/x is the value before physics step <step>; post/x is the offending '
                'post-step state; last_finite/* are contacts/constraints of the last finite step.',
    }, dtype=object)
    guard.output_dir.mkdir(parents=True, exist_ok=True)
    path = guard.output_dir / f'nan_forensics_{datetime.now():%Y%m%d_%H%M%S}.npz'
    np.savez_compressed(path, **out)
    print(f'[NanForensics] Wrote {len(out)} arrays to {path}', flush=True)
    return path
