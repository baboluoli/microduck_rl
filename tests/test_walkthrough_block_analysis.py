"""Offline prefix semantics; these tests never import the simulator."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'block_analysis', Path(__file__).resolve().parents[1] / 'scripts/analyze_walkthrough_blocks.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

COLUMNS = ['t', 'x', 'y', 'vx', 'vz', 'tilt_deg', 'left_support', 'right_support',
           'head_force', 'trunk_force', 'bad_impact', 'lower_floor',
           'walk_milestone_step', 'walk_blocks', 'walk_bad', 'walk_pulse']


def trace(*, steps=600, failure=None, failure_step=166, no_landing=False):
    """A 0.1 m/s walk; milestone is step 16, first block ends at step 66."""
    age = np.arange(1, steps + 1)
    c = {name: np.zeros(steps) for name in COLUMNS}
    c.update(t=age * .02, x=.95 + age * .002, vx=np.full(steps, .1),
             left_support=np.ones(steps), lower_floor=np.ones(steps),
             walk_milestone_step=np.where(age >= 16, 16, -1),
             walk_blocks=np.maximum((age - 16) // 50, 0))
    if no_landing:
        c['lower_floor'][:] = 0
        c['walk_milestone_step'][:] = -1
        c['walk_blocks'][:] = 0
    if failure:
        c['walk_bad'][age >= failure_step] = 1
        i = failure_step - 1
        if failure == 'progress':
            # Fail block 3, then resume progress. Later valid blocks must not
            # count toward a new prefix.
            c['x'][i] = c['x'][i - 50]
        elif failure == 'head':
            c['head_force'][i] = 15.01
        elif failure == 'trunk':
            c['trunk_force'][i] = 20.01
        elif failure == 'tilt':
            c['tilt_deg'][i] = 45
        elif failure == 'boundary':
            c['y'][i] = 1.16
        elif failure == 'nonfinite':
            c['vx'][i] = np.nan
        else:
            raise ValueError(failure)
    if steps == 600 and not failure and not no_landing:
        c['walk_pulse'][-1] = 1
    return np.stack([c[name] for name in COLUMNS], axis=-1)


@pytest.mark.parametrize('failure', ['progress', 'head', 'trunk', 'tilt', 'boundary', 'nonfinite'])
def test_first_failure_latches_and_vetoes_block_ending_on_same_sample(failure):
    result = analysis.classify_prefix(trace(failure=failure), COLUMNS)
    assert result['clean_prefix_blocks'] == 2
    assert result['observed_blocks'] == 11
    assert not result['strict_success']
    assert result['first_failure']['time_s'] == pytest.approx(3.32)


def test_later_failure_keeps_earned_blocks_but_loses_strict_success():
    result = analysis.classify_prefix(trace(failure='tilt', failure_step=590), COLUMNS)
    assert result['clean_prefix_blocks'] == 11
    assert not result['strict_success']


def test_partial_block_never_pays():
    result = analysis.classify_prefix(trace(steps=115), COLUMNS)
    assert result['clean_prefix_blocks'] == 1
    assert not result['full_horizon']
    assert not result['strict_success']


def test_no_lower_floor_landing_never_counts_walking():
    result = analysis.classify_prefix(trace(no_landing=True), COLUMNS)
    assert result['milestone_time_s'] is None
    assert result['clean_prefix_blocks'] == 0


def test_valid_full_episode_and_budget_cap():
    result = analysis.classify_prefix(trace(), COLUMNS)
    assert result['strict_success']
    assert result['clean_prefix_blocks'] == 11
    summary = analysis.summarize([dict(termination='time_out', **result)])
    assert summary['proposed_bonus_total'] == 80


def test_missing_sample_rejected_instead_of_counting_a_gap():
    with pytest.raises(ValueError, match='contiguous'):
        analysis.classify_prefix(np.delete(trace(), 100, axis=0), COLUMNS)
