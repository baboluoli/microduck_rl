"""Height ladder to 10 cm: bank, stage sampler, curriculum gate and session loop (CPU)."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks.cliff import course
from mjlab_microduck.tasks.cliff.course import HEIGHTS, STAGES, bands, reset
from mjlab_microduck.tasks.cliff.course_curriculum import Curriculum, Result

SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))  # The runner imports cliff_course_process lazily, from scripts/.
spec = importlib.util.spec_from_file_location('height_session', SCRIPTS / 'run_height_curriculum_session.py')
session = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session)

OLD_CURRICULUM = Path(__file__).parents[1] / 'logs/course-session-20261024/artifacts/logs/course-session/curriculum.json'
PASS, FAIL = Result((22, 22, 22), 0, 72), Result((15, 15, 15), 0, 60)


def test_bank_keeps_old_rows_and_contains_every_stage():
    assert len(HEIGHTS) == 49
    assert HEIGHTS[:21] == tuple((40 + i) / 2000 for i in range(21))  # Original rows unchanged.
    assert all(s in HEIGHTS for s in STAGES) and HEIGHTS[-1] == STAGES[-1] == .10
    assert all(b > a for a, b in zip(HEIGHTS, HEIGHTS[1:]))
    # Values pass through argparse (str -> float) and mm/1000 exactly.
    assert all(float(str(s)) in HEIGHTS and round(s * 1000, 1) / 1000 in STAGES for s in STAGES)


def _env(n):
    terrain = SimpleNamespace(terrain_levels=torch.zeros(n, dtype=torch.long), terrain_types=torch.zeros(n, dtype=torch.long),
        env_origins=torch.zeros(n, 3), terrain_origins=torch.zeros(len(HEIGHTS), 1, 3))
    robot = SimpleNamespace(data=SimpleNamespace(indexing=SimpleNamespace(free_joint_q_adr=list(range(7)))),
                            write_root_link_pose_to_sim=lambda pose, env_ids: None)
    scene = type('Scene', (), {'terrain': terrain, 'env_origins': terrain.env_origins,
                               '__getitem__': lambda self, key: robot})()
    return SimpleNamespace(scene=scene, device='cpu', sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(n, 7))))


@pytest.mark.parametrize('upper', STAGES)
def test_sampler_bands_per_stage(upper):
    n = 20000
    env = _env(n)
    torch.manual_seed(0)
    reset(env, torch.arange(n), upper=upper)
    h = torch.tensor(HEIGHTS)[env.scene.terrain.terrain_levels]
    stage = STAGES.index(upper)
    if stage == 0:
        assert (h == .02).all()
        return
    previous = STAGES[stage - 1]
    assert (h <= upper + 1e-9).all()
    frontier = h > previous + 1e-9
    assert frontier.float().mean().item() == pytest.approx(.75, abs=.015)
    # Every row of each band is actually used.
    easier_end, frontier_end = bands(upper)
    assert set(env.scene.terrain.terrain_levels.tolist()) == set(range(frontier_end))


def test_stage_2_25_draws_exactly_the_original_rows():
    def original(n, upper):  # The 2–3 cm sampler as it was on 2026-10-24.
        top = round((upper - .02) * 2000)
        frontier = torch.randint(max(1, top - 4), max(2, top + 1), (n,))
        easier = torch.randint(0, max(1, top - 4), (n,))
        return torch.where(torch.rand(n) < .75, frontier, easier) if top else torch.zeros_like(easier)
    for upper in (.02, .0225, .025):
        torch.manual_seed(7)
        expected = original(4096, upper)
        env = _env(4096)
        torch.manual_seed(7)
        reset(env, torch.arange(4096), upper=upper)
        assert torch.equal(env.scene.terrain.terrain_levels, expected)


def test_unsupported_heights_rejected():
    env = _env(4)
    with pytest.raises(ValueError):
        reset(env, torch.arange(4), upper=.0275)  # Old 2.5 mm ladder value, not a stage.
    with pytest.raises(ValueError):
        reset(env, torch.arange(4), fixed_height=.0305)


def _all(c, result):
    return {k: result for k in c.conditions()}


def test_curriculum_climbs_stages_in_order_and_stops_at_10cm():
    c = Curriculum()
    seen = [c.upper_mm]
    while True:
        dev = _all(c, PASS)
        confirm = {k: PASS for k in c.frontier()}
        if not c.advance(dev, confirm, f'ckpt-{c.upper_mm:g}'):
            break
        seen.append(c.upper_mm)
    assert [s / 1000 for s in seen] == list(STAGES)
    assert c.upper_mm == 100 and c.mastered and c.last_stage
    assert not c.advance(_all(c, PASS), {k: PASS for k in c.frontier()}, 'again')
    assert c.upper_mm == 100


def test_advance_needs_both_seed_sets_and_ignores_regression():
    c = Curriculum(upper_mm=30)
    assert c.conditions().keys() == {'30/45', '30/55', '30/65', '20/55', '25/55'}
    dev = _all(c, PASS)
    assert not c.advance(dev, {k: (FAIL if k == '30/65' else PASS) for k in c.frontier()}, 'x')
    assert not c.advance({**dev, '30/45': FAIL}, {k: PASS for k in c.frontier()}, 'x')
    with pytest.raises(ValueError):
        c.advance(dev, {'30/45': PASS}, 'x')
    c.blocks_at_stage = 3
    assert c.advance({**dev, '20/55': FAIL}, {k: PASS for k in c.frontier()}, 'x')  # Regression is checked separately.
    assert c.upper_mm == 35 and c.blocks_at_stage == 0 and c.history[-1]['blocks'] == 3


def test_old_curriculum_file_loads():
    if not OLD_CURRICULUM.exists():
        pytest.skip('Gitignored 2026-10-24 session artifact not present')
    c = Curriculum.load(OLD_CURRICULUM)
    assert c.upper_mm == 22.5 and c.stage == 1 and c.blocks_at_stage == 0
    assert list(c.conditions()) == ['22.5/45', '22.5/55', '22.5/65', '20/55']
    # Streaks under the old success-drop rule are kept as a record, not reused.
    assert c.streaks and not c.regression_streaks
    assert not c.regression(_all(c, FAIL))  # First failure: streak 1, no stop yet.


def test_regression_streak_uses_only_regression_conditions(tmp_path):
    c = Curriculum(upper_mm=30)
    frontier_fail = {**_all(c, PASS), **{k: FAIL for k in c.frontier()}}
    assert not c.regression(frontier_fail) and not c.regression(frontier_fail)
    one = {**_all(c, PASS), '25/55': FAIL}
    assert not c.regression(one)
    c.save(tmp_path / 'c.json')
    c = Curriculum.load(tmp_path / 'c.json')
    assert not c.regression(_all(c, PASS))  # A pass resets the streak.
    assert not c.regression(one)
    assert c.regression(one)  # Second consecutive failure stops.
    # After an advance the previous stage changes; its old streak is dropped.
    c = Curriculum(upper_mm=30, regression_streaks={'25/55': 1})
    c.advance(_all(c, PASS), {k: PASS for k in c.frontier()}, 'x')
    assert not c.regression({**_all(c, PASS), '25/55': FAIL})
    assert '25/55' not in c.regression_streaks


def test_sampled_bands_from_counts():
    counts = [0] * len(HEIGHTS)
    counts[0], counts[3] = 25, 75
    assert session.sampled_bands(counts, .0225) == dict(frontier_share=.75, above_stage=0, total=100)
    counts[30] = 1
    assert session.sampled_bands(counts, .0225)['above_stage'] == 1


class FakeSession:
    """Stubbed train/evaluate with a fake clock: one block costs 4.5 min, checks 8 and 5."""

    def __init__(self, state, outcomes, start=0., deadline=10**6):
        self.state, self.outcomes, self.t, self.deadline = state, iter(outcomes), start, deadline
        self.records, self.trained, self.warm = [], [], []

    def train_block(self, block, warm):
        self.t += 270
        self.trained.append(self.state.upper_mm)
        self.warm.append(warm)
        self.current = next(self.outcomes)
        return Path(f'model_{block * 100 - 1}.pt')

    def check(self, checkpoint):
        self.t += 480
        frontier_ok, regression_ok = self.current[:2]
        return {**{k: PASS if frontier_ok else FAIL for k in self.state.frontier()},
                **{k: PASS if regression_ok else FAIL for k in self.state.regression_conditions()}}

    def confirm(self, checkpoint):
        self.t += 300
        return {k: PASS if self.current[2] else FAIL for k in self.state.frontier()}

    def record(self, entry, checkpoint, passed_stage):
        json.dumps(entry)  # Must be serializable for stages.jsonl.
        self.records.append((entry, passed_stage))

    def run(self, **kwargs):
        return session.run_blocks(self.state, train_block=self.train_block, check=self.check, confirm=self.confirm,
                                  now=lambda: self.t, deadline=self.deadline, record=self.record, **kwargs)


def test_loop_advances_then_stalls():
    # (frontier passes on development, regression passes, confirmation passes)
    outcomes = [(True, True, True), (False, True, None), (True, True, False)] + [(False, True, None)] * 4
    fake = FakeSession(Curriculum(upper_mm=22.5), outcomes)
    assert fake.run() == 'stall'
    assert fake.trained == [22.5] + [25] * 5  # Block 1 mastered 2.25 cm; 2.5 cm then failed 5 blocks.
    assert fake.warm == [True] + [False] * 5
    assert fake.records[0][0]['advanced'] and fake.records[0][1] == 22.5
    assert fake.records[2][0]['confirmation'] is not None and not fake.records[2][0]['advanced']
    assert all(p is None for _, p in fake.records[1:])
    assert fake.state.blocks_at_stage == 5 and fake.state.upper_mm == 25


def test_loop_stops_on_two_regressions():
    fake = FakeSession(Curriculum(upper_mm=25), [(False, False, None), (True, False, None)])
    assert fake.run() == 'regression'
    assert len(fake.trained) == 2
    # Regression on the stop check: confirmation is skipped, the stage is not raised.
    assert fake.records[-1][0]['confirmation'] is None and fake.state.upper_mm == 25


def test_loop_respects_time_budget():
    # 17 min per block: with 40 min left, two blocks start, a third does not.
    fake = FakeSession(Curriculum(upper_mm=22.5), [(True, True, True)] * 10, deadline=40 * 60)
    assert fake.run() == 'time budget'
    assert fake.trained == [22.5, 25]
    assert fake.deadline - fake.t < session.BLOCK_BUDGET_S


def test_loop_stops_when_last_stage_mastered():
    fake = FakeSession(Curriculum(upper_mm=90), [(True, True, True)] * 3)
    assert fake.run() == 'last stage mastered'
    assert fake.trained == [90, 100]
    assert [p for _, p in fake.records] == [90, 100]


def test_hold_mode_never_advances_and_runs_exactly_n_blocks():
    # Every gate passes, so the normal loop would climb a stage per block.
    fake = FakeSession(Curriculum(upper_mm=22.5), [(True, True, True)] * 10)
    assert fake.run(hold=True, max_blocks=4) == 'block limit'
    assert fake.trained == [22.5] * 4 and fake.state.upper_mm == 22.5
    assert fake.state.history == [] and fake.state.blocks_at_stage == 4
    assert all(e['would_advance'] and not e['advanced'] and e['next_upper_mm'] == 22.5 for e, _ in fake.records)
    assert [p for _, p in fake.records] == [22.5] * 4  # Each passing block is a ``best`` candidate.


def test_hold_mode_has_no_stall_stop_and_records_would_advance():
    # Frontier fails 6 times (normal mode stalls after 5), then confirmation fails once.
    outcomes = [(False, True, None)] * 6 + [(True, True, False)]
    fake = FakeSession(Curriculum(upper_mm=22.5), outcomes)
    assert fake.run(hold=True, max_blocks=7) == 'block limit'
    assert len(fake.trained) == 7
    assert not any(e['would_advance'] for e, _ in fake.records)
    assert fake.records[-1][0]['confirmation'] is not None and all(p is None for _, p in fake.records)


def test_hold_mode_still_stops_on_regression():
    fake = FakeSession(Curriculum(upper_mm=22.5), [(True, False, None), (True, False, None)] + [(True, True, True)] * 2)
    assert fake.run(hold=True, max_blocks=4) == 'regression'
    assert len(fake.trained) == 2 and fake.state.upper_mm == 22.5


def test_walking_quality_from_synthetic_trace(tmp_path):
    from summarize_height_curriculum import action_std, walking_quality

    def sample(x, age, vx, yaw):
        return dict(x=x, age=age, finite=True, velocity_body=[vx, 0., 0.], angular_velocity_body=[0., 0., yaw])
    # Episode 1: young samples (age <= 10) and the edge zone (0.5-0.8 m) are ignored.
    one = [sample(0., 5, 9., 9.), sample(.1, 11, .2, -1.), sample(.4, 12, .4, 1.), sample(.6, 13, 9., 9.),
           sample(.9, 14, .3, 0.), sample(1., 15, .5, 0.)]
    two = [sample(.2, 11, .6, 2.), sample(.9, 12, .7, 0.), dict(sample(1., 13, 99., 99.), finite=False)]
    (tmp_path / 'h0.0225-d0.55-s20.json').write_text(json.dumps([
        dict(trace=one, success=True, duration=4.), dict(trace=two, success=False, duration=1.)]))
    q = walking_quality(tmp_path)
    # Per-episode means first: before (0.3, 0.6), after (0.4, 0.7), |yaw| (1, 2).
    assert q == pytest.approx(dict(speed_before_edge=.45, speed_after_landing=.55,
                                   yaw_rate_before_edge=1.5, time_to_goal_s=4.))
    assert action_std('Mean action std: 1.05\n...\n  Mean action std: 1.19\n') == 1.19
    assert action_std('no training output') is None


def test_course_entropy_bonus_is_reduced_only_for_course():
    from mjlab_microduck.tasks.cliff.env_cfg import MicroduckCliffCourseRlCfg, MicroduckCliffOracleRlCfg
    assert MicroduckCliffCourseRlCfg.algorithm.entropy_coef == .003
    assert MicroduckCliffOracleRlCfg.algorithm.entropy_coef == .01  # Historical tasks unchanged.


def _walk(speeds, cap=.21*.02, dt=.02, contact_at=.65, bad_at=None):
    """Drive CourseState along X at the given per-step speeds; return (total points, success)."""
    from mjlab_microduck.tasks.cliff.course import CourseState, POINTS_PER_METRE
    s, x, total = CourseState(1, 'cpu'), contact_at, 0.
    for step, v in enumerate(speeds, 1):
        x += v * dt
        head = 16. if bad_at is not None and step == bad_at else 0.
        pulse = s.update(step=step, age=torch.tensor([step]), x=torch.tensor([x]), y=torch.zeros(1),
                         lower=torch.tensor([True]), cos_tilt=torch.ones(1), head=torch.tensor([head]),
                         trunk=torch.zeros(1), finite=torch.tensor([True]), failure=torch.tensor([False]),
                         cap=torch.tensor([cap]))
        total += POINTS_PER_METRE * s.gain.item()
        if pulse.item():
            return total, True
    return total, False


def test_completion_credit_pays_commanded_pace_best():
    from mjlab_microduck.tasks.cliff.course import COMPLETION_POINTS
    steady, ok = _walk([.21] * 2000)
    assert ok and steady == pytest.approx(COMPLETION_POINTS, rel=1e-3)  # The full metre is paid.
    rushed, ok = _walk([.5] * 2000)
    assert ok and rushed < .5 * steady  # Arriving at 2.4x the command forfeits the unpaid backlog.
    # A brief speed burst followed by slowing down is paid in full via the backlog.
    burst, _ = _walk([.4] * 50 + [.1] * 2000)
    assert burst == pytest.approx(steady, rel=1e-2)


def test_completion_credit_ignores_retreat_and_stops_on_impact():
    shuffle, _ = _walk([.3, -.3] * 500)  # Back and forth: only new ground counts.
    assert shuffle == pytest.approx(_walk([.3])[0], rel=1e-6)
    clean, _ = _walk([.21] * 100)
    hit, ok = _walk([.21] * 100, bad_at=50)
    # Step 1 is the touchdown (pays nothing); steps 2-49 pay, the impact step does not.
    assert not ok and hit == pytest.approx(clean * 48 / 99, rel=1e-6)
