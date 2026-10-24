"""Distance-goal semantics without importing Torch, MuJoCo or a GPU backend."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'distance_analysis', Path(__file__).resolve().parents[1]/'scripts/analyze_cliff_distance.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
COLUMNS = ['t','x','y','vx','tilt_deg','head_force','trunk_force','bad_impact','lower_floor']


def trace():
    # Genuine lower-floor contact at X=.6; the one-metre goal is X=1.6.
    values = np.zeros((10,len(COLUMNS)))
    c = {name:i for i,name in enumerate(COLUMNS)}
    values[:,c['t']] = np.arange(1,11)*.02
    values[:,c['x']] = [.0,.4,.6,.8,1.,1.2,1.4,1.6,1.8,2.]
    values[:,c['vx']] = 10.
    values[2,c['lower_floor']] = 1
    return values


def change(values, row, **fields):
    for name,value in fields.items():
        values[row,COLUMNS.index(name)] = value
    return values


def test_success_uses_first_contact_and_net_distance_not_elapsed_time_or_old_hold():
    result = analysis.classify(trace(), COLUMNS)
    assert result['success']
    assert result['first_lower_contact']['root_x_m'] == .6
    assert result['completion']['step'] == 8
    assert result['completion']['progress_m'] == pytest.approx(1.)


@pytest.mark.parametrize('fields', [
    {'head_force':15.01}, {'trunk_force':20.01}, {'tilt_deg':45.},
    {'y':1.16}, {'vx':np.nan},
])
def test_goal_sample_failure_vetoes_success_permanently(fields):
    result = analysis.classify(change(trace(),7,**fields), COLUMNS)
    assert not result['success']
    assert result['failure_before_goal']
    assert result['first_failure']['step'] == 8


def test_contact_sample_tilt_veto_and_no_later_reanchoring():
    values = change(trace(),2,tilt_deg=45.)
    values = change(values,4,lower_floor=1.)
    result = analysis.classify(values,COLUMNS)
    assert not result['success']
    assert result['first_lower_contact']['step'] == 3


def test_precontact_impact_is_not_erased_by_landing():
    assert not analysis.classify(change(trace(),1,head_force=16.),COLUMNS)['success']


@pytest.mark.parametrize('fields',[{'head_force':16.},{'tilt_deg':60.},{'x':3.81}])
def test_later_invalidity_cannot_undo_completed_task(fields):
    result = analysis.classify(change(trace(),8,**fields),COLUMNS)
    assert result['success'] and result['later_failure_after_success']
    assert not result['failure_before_goal']


def test_genuine_contact_required_and_airborne_steps_do_not_restart_distance():
    values = trace()
    values[:,COLUMNS.index('lower_floor')] = 0
    assert not analysis.classify(values,COLUMNS)['success']
    # trace() has exactly one contact sample, which is enough to anchor it.
    assert analysis.classify(trace(),COLUMNS)['success']


def test_path_length_cannot_replace_net_forward_progress():
    values = trace()
    values[:,COLUMNS.index('x')] = [.0,.4,.6,1.2,.6,1.2,.6,1.2,.6,1.2]
    assert not analysis.classify(values,COLUMNS)['success']


def test_partial_observation_and_missing_samples():
    assert not analysis.classify(trace()[:7],COLUMNS)['success']
    with pytest.raises(ValueError, match='contiguous'):
        analysis.classify(np.delete(trace(),4,axis=0),COLUMNS)


def test_summaries_use_all_attempts_and_keep_post_goal_events_separate():
    good=analysis.classify(trace(),COLUMNS)
    bad=analysis.classify(change(trace(),4,tilt_deg=45.),COLUMNS)
    for row in (good,bad):
        row['termination']='time_out'
    result=analysis.summarize([good,bad])
    assert result['attempts']==2 and result['success_rate']==.5


def test_later_boundary_is_distinguished_from_a_physical_failure():
    score=analysis.classify(change(trace(),8,x=3.81),COLUMNS)
    score['termination']='diagnostic_tile_boundary'
    summary=analysis.summarize([score])
    assert summary['successes_with_later_boundary']==1
    assert summary['successes_with_later_physical_invalidity']==0
    assert summary['observed_events_before_goal_or_end']['boundary']==0


def test_pre_goal_events_include_impact_after_earlier_tilt_failure():
    values=change(trace(),4,tilt_deg=45.)
    values=change(values,6,head_force=16.)
    score=analysis.classify(values,COLUMNS)
    score['termination']='cliff_failed_landing'
    summary=analysis.summarize([score])
    assert summary['first_disqualification_before_goal']['tilt']==1
    assert summary['first_disqualification_before_goal']['impact']==0
    assert summary['observed_events_before_goal_or_end']['impact']==1
