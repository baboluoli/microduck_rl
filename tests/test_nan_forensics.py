"""CPU regression tests for the extended NaN-guard evidence (no GPU, no stepping)."""
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from mjlab_microduck import nan_forensics

XML = """<mujoco><worldbody><body><freejoint/><geom size='.1'/></body>
<body pos='1 0 0'><joint type='hinge'/><geom size='.1'/></body></worldbody>
<actuator><motor joint='unnamed_joint_0' name='m'/></actuator></mujoco>"""
XML = XML.replace("joint='unnamed_joint_0'", "joint='h'").replace("type='hinge'", "name='h' type='hinge'")


def fake_data(n, nv, nu, step):
    contact = SimpleNamespace(
        dist=torch.tensor([-.001, -.005, .0, -.002]), pos=torch.zeros(4, 3), frame=torch.zeros(4, 3, 3),
        friction=torch.zeros(4, 5), solref=torch.zeros(4, 2), solimp=torch.zeros(4, 5),
        dim=torch.full((4,), 3), geom=torch.tensor([[0, 1], [0, 2], [1, 2], [0, 1]]),
        efc_address=torch.zeros(4, 4, dtype=torch.long), includemargin=torch.zeros(4),
        worldid=torch.tensor([0, 1, 1, 2]))
    efc = SimpleNamespace(**{k: torch.full((n, 8), float(step)) for k in nan_forensics.EFC_FIELDS})
    return SimpleNamespace(
        qpos=torch.full((n, nv + 1), float(step)), qvel=torch.full((n, nv), float(step)),
        qacc=torch.zeros(n, nv), qacc_warmstart=torch.zeros(n, nv), ctrl=torch.full((n, nu), step + .5),
        actuator_force=torch.zeros(n, nu), qfrc_actuator=torch.zeros(n, nv), qfrc_constraint=torch.zeros(n, nv),
        qfrc_applied=torch.zeros(n, nv), qfrc_bias=torch.zeros(n, nv), qfrc_passive=torch.zeros(n, nv),
        sensordata=torch.zeros(n, 0), solver_niter=torch.full((n,), 3), nefc=torch.tensor([2, 5, 1]),
        time=torch.zeros(n), xpos=torch.zeros(n, 3, 3), xquat=torch.zeros(n, 3, 4), act=torch.zeros(n, 0),
        nacon=torch.tensor([3]), contact=contact, efc=efc)


def test_guard_writes_pre_post_contacts_actions_and_model(tmp_path):
    from mjlab.utils.nan_guard import NanGuard, NanGuardCfg
    nan_forensics.install()
    nan_forensics.install()  # Idempotent.
    model = mujoco.MjModel.from_xml_string(XML)
    n = 3
    guard = NanGuard(NanGuardCfg(enabled=True, output_dir=str(tmp_path), buffer_size=4), n, model)
    # What the patched Simulation.__init__ attaches.
    from collections import deque
    guard.extra, guard.actions, guard.last_contacts = deque(maxlen=4), deque(maxlen=4), None
    friction = torch.arange(n, dtype=torch.float32)
    actuator = SimpleNamespace(friction_scale=friction)
    guard.env = SimpleNamespace(
        scene=SimpleNamespace(entities={'robot': SimpleNamespace(actuators=[actuator])}, env_origins=torch.zeros(n, 3)),
        episode_length_buf=torch.tensor([5, 6, 7]), common_step_counter=11)
    guard.actions.append({'env_step': 10, 'physics_step': 0, 'action': torch.ones(n, model.nu)})
    for step in range(6):
        guard.capture(fake_data(n, model.nv, model.nu, step))
    post = fake_data(n, model.nv, model.nu, 99)
    post.qpos[1, 0] = float('nan')
    guard.sim = SimpleNamespace(data=post, model=SimpleNamespace(geom_friction=torch.ones(n, 2, 3)),
                                expanded_fields={'geom_friction'})
    assert guard.check_and_dump(post)
    files = sorted(tmp_path.glob('nan_forensics_*.npz'))
    assert len(files) == 1 and list(tmp_path.glob('nan_dump_*.npz'))
    data = np.load(files[0], allow_pickle=True)
    keys = set(data.files)
    # Buffer of 4: steps 2..5 retained, values from the pre-step state.
    assert {f'pre/{s:06d}/ctrl' for s in range(2, 6)} <= keys
    assert data['pre/000005/ctrl'].shape == (1, model.nu) and data['pre/000005/ctrl'][0, 0] == 5.5
    assert np.isnan(data['post/qpos'][0, 0])
    assert data['action/000010/action'].shape == (1, model.nu)
    # Env 1 owns contacts 1 and 2 of the 3 active ones; the 4th is beyond nacon.
    np.testing.assert_allclose(data['post/env1/contact_dist'], [-.005, 0.])
    np.testing.assert_allclose(data['last_finite/env1/contact_dist'], [-.005, 0.])
    assert data['post/env1/efc_force'].shape == (5,)
    assert data['model/geom_friction'].shape == (1, 2, 3)
    assert data['episode_length_buf'].tolist() == [6]
    meta = data['_metadata'].item()
    assert meta['dumped_env_ids'] == [1]


class FakeActuator:
    pass


FakeActuator.__module__ = 'mjlab_microduck.actuator.fake'


def test_entity_scan_reads_per_env_tensors_on_mjlab_objects():
    actuator = FakeActuator()
    actuator.friction_scale = torch.tensor([1., 2., 3.])
    actuator.shared = torch.ones(5)  # Not per-environment: ignored.
    env = SimpleNamespace(scene=SimpleNamespace(entities={'robot': actuator}))
    out = nan_forensics.entity_tensors(env, 3, [2])
    assert out == {'env2/entity.robot.friction_scale': 3.}


def test_disabled_guard_is_untouched(tmp_path):
    from mjlab.utils.nan_guard import NanGuard, NanGuardCfg
    nan_forensics.install()
    guard = NanGuard(NanGuardCfg(enabled=False, output_dir=str(tmp_path)), 2, mujoco.MjModel.from_xml_string(XML))
    guard.capture(None)  # Must not touch data when disabled.
    assert not hasattr(guard, 'extra')
