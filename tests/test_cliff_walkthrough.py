"""Synthetic control-rate traces: reward timing and independent episode state."""
import math
import torch
import pytest
from mjlab_microduck.tasks.cliff.walkthrough import WalkingState, lower_floor_contact
from mjlab_microduck.tasks.cliff.env_cfg import make_microduck_cliff_walkthrough_env_cfg


def tick(s, i, *, x=None, **overrides):
    n = len(s.bad)
    args = dict(step=i, age=torch.full((n,), i), x=torch.full((n,), .95 + i*.002),
                y=torch.zeros(n), vx=torch.full((n,), .1), vz=torch.zeros(n),
                cos_tilt=torch.ones(n), supported=torch.ones(n, dtype=torch.bool),
                lower=torch.ones(n, dtype=torch.bool), head=torch.zeros(n), trunk=torch.zeros(n),
                finite=torch.ones(n, dtype=torch.bool), horizon=600)
    if x is not None:
        args['x'] = torch.full((n,), x)
    args.update(overrides)
    return s.update(**args)


@pytest.mark.parametrize('play', [False, True])
def test_cfg(play):
    cfg = make_microduck_cliff_walkthrough_env_cfg(play)
    assert cfg.episode_length_s == 12
    assert 'cliff_landing' not in cfg.terminations
    assert cfg.events['reset_base'].params['pose_range']['x'] == (0., 0.)
    assert cfg.events['cliff_midair_reset'].params['probability'] == 0
    assert cfg.rewards['cliff_landing'].weight == 1
    assert cfg.observations['actor'].terms['cliff_terrain'].params['drop_height'] == .02


def test_valid_timeout_and_cached_calls():
    s = WalkingState(2, 'cpu', .02)
    for i in range(1, 600):
        assert not tick(s, i).any()
        assert not tick(s, i).any()
    assert (s.milestone_step == 16).all()
    assert tick(s, 600).all()
    assert tick(s, 600).all()  # same pulse for another manager; no extra update
    assert not tick(s, 601).any()
    assert (s.blocks == 11).all()


@pytest.mark.parametrize('mode', ['stand', 'backward', 'tilt', 'head', 'trunk', 'boundary', 'nan', 'no_contact', 'late', 'later_tilt'])
def test_invalid(mode):
    s = WalkingState(1, 'cpu', .02)
    for i in range(1, 601):
        kw = {}
        if mode == 'stand': kw['x'] = .96
        if mode == 'backward': kw['x'] = 2. - i*.001
        if mode in ('tilt', 'later_tilt') and i == (100 if mode == 'tilt' else 590):
            kw['cos_tilt'] = torch.tensor([math.cos(math.pi/4)])
        if mode in ('head', 'trunk') and i == 100: kw[mode] = torch.tensor([21.])
        if mode == 'boundary' and i == 600: kw['y'] = torch.tensor([1.2])
        if mode == 'nan' and i == 100: kw['finite'] = torch.tensor([False])
        if mode == 'no_contact' or (mode == 'late' and i < 400): kw['lower'] = torch.tensor([False])
        assert not tick(s, i, **kw).any()


def test_alternating_aerial_phases_do_not_clear_blocks_and_partial_reset():
    s = WalkingState(2, 'cpu', .02)
    for i in range(1, 601):
        kw = {'supported': torch.zeros(2, dtype=torch.bool)} if i > 16 else {}
        tick(s, i, **kw)
    assert s.pulse.all()
    s.reset(torch.tensor([0]))
    assert s.blocks.tolist() == [0, 11]
    assert s.milestone_step.tolist() == [-1, 16]
    assert s.paid.tolist() == [False, True]


def test_real_lower_contact():
    found = torch.ones(4, 1, dtype=torch.bool)
    pos = torch.tensor([[[1., 0, -.02]], [[1., 0, 0]], [[.3, 0, -.02]], [[4.1, 0, -.02]]])
    assert lower_floor_contact(found, pos, torch.zeros(4, 3)).tolist() == [True, False, False, False]


def test_independent_scalar_classifier_matches_vectorized_state():
    import sys
    from pathlib import Path
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import eval_cliff_walkthrough as ev
    original = ev.diagnostic.COLUMNS
    ev.diagnostic.COLUMNS = original + ev.EXTRA
    try:
        for mode in ('good', 'stand', 'late', 'impact', 'boundary', 'later_tilt'):
            s = WalkingState(1, 'cpu', .02)
            rows = []
            for i in range(1, 601):
                x = .95 + i*.002 if mode != 'stand' else .96
                head = 21. if mode == 'impact' and i == 200 else 0.
                tilt = 46. if mode == 'later_tilt' and i == 590 else 0.
                y = 1.2 if mode == 'boundary' and i == 600 else 0.
                lower = mode != 'late' or i >= 400
                tick(s, i, x=x, y=torch.tensor([y]), head=torch.tensor([head]),
                     cos_tilt=torch.tensor([math.cos(math.radians(tilt))]), lower=torch.tensor([lower]))
                row = {name:0. for name in ev.diagnostic.COLUMNS}
                row.update(t=i*.02, x=x, y=y, vx=.1, tilt_deg=tilt, left_support=1.,
                           head_force=head, bad_impact=float(s.impact[0]), lower_floor=float(lower))
                rows.append([row[c] for c in ev.diagnostic.COLUMNS])
            scored = ev.classify(np.array(rows))
            assert scored['success'] == bool(s.pulse[0]), mode
            assert scored['walking_blocks'] == int(s.blocks[0]), mode
    finally:
        ev.diagnostic.COLUMNS = original


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA runtime preflight runs on the pod')
@pytest.mark.parametrize('play', [False, True])
def test_cuda_interface_and_fixed_reset(play):
    from mjlab.envs import ManagerBasedRlEnv
    cfg = make_microduck_cliff_walkthrough_env_cfg(play)
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        for seed in (42, 43):
            env.reset(seed=seed)
            p = env.scene['robot'].data.root_link_pos_w-env.scene.env_origins
            torch.testing.assert_close(p[:, 0], torch.zeros(4, device='cuda:0'), atol=2e-5, rtol=0)
            assert env.observation_manager.group_obs_dim['actor'] == (63,)
            assert env.observation_manager.group_obs_dim['critic'] == (78,)
            assert env.action_manager.total_action_dim == 14
    finally:
        env.close()


def test_missed_timeout_cannot_pay_late():
    s = WalkingState(1, 'cpu', .02)
    for i in range(1, 701):
        assert not tick(s, i, lower=torch.tensor([i >= 400])).any()
    assert s.blocks.item() >= 5


@pytest.mark.parametrize('failure_step,expected', [(0,80),(66,0),(67,8),(266,32),(267,40),(600,40)])
def test_partial_points_veto_cap_and_retained_credit(failure_step, expected):
    from types import SimpleNamespace
    from mjlab_microduck.tasks.cliff.walkthrough import reward
    s = WalkingState(1, 'cpu', .02)
    env = SimpleNamespace(_walkthrough=s, step_dt=.02, reset_terminated=torch.tensor([False]))
    total = 0
    for i in range(1, 602):
        tick(s, i, head=torch.tensor([21. if i == failure_step else 0.]))
        env.common_step_counter = i
        # Calls by different managers see identical evidence and pulses.
        value = reward(env)
        torch.testing.assert_close(value, reward(env))
        total += value.item() * env.step_dt
    assert total == pytest.approx(expected)
    assert s.paid_blocks.item() == min(expected // 8, 5)


def test_timeout_can_pay_block_and_final_together():
    s = WalkingState(1, 'cpu', .02)
    for i in range(1,601):
        tick(s, i, lower=torch.tensor([i>=350]))
    assert s.pulse.item() and s.block_pulse.item()
    assert s.paid_blocks.item() == 5
    s.reset(torch.tensor([0]))
    assert s.paid_blocks.item() == 0 and s.clean_blocks.item() == 0


def test_staggered_partial_payment_rows():
    s = WalkingState(2, 'cpu', .02)
    for i in range(1,201):
        tick(s,i)
    assert s.paid_blocks.tolist() == [3,3]
    for i in range(201,267):
        tick(s,i,age=torch.tensor([i-200,i]))
    assert s.paid_blocks.tolist() == [1,5]
    assert s.clean_blocks.tolist() == [1,5]
