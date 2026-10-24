"""Guard the diagnostic against success censoring and misleading controls."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0,str(scripts))
import diagnose_cliff_oracle as diagnostic


def trace(n=11):
    values=np.zeros((n,len(diagnostic.COLUMNS)))
    values[:,diagnostic.COLUMNS.index('t')]=np.arange(n)
    values[2:,diagnostic.COLUMNS.index('x')]=.6
    return values


def test_goal_does_not_hide_later_failure():
    values=trace()
    values[3,diagnostic.COLUMNS.index('goal_pulse')]=1
    values[5:,diagnostic.COLUMNS.index('bad_impact')]=1
    result=diagnostic.describe_trace(values,termination='time_out',physical_cliff=True)
    assert result['original_goal_time_s']==3
    assert result['original_goal_then_failure']
    assert result['edge_plus_5s_clean'] is False
    assert result['goal_plus_5s_clean'] is False


def test_short_failed_rollout_is_failure_not_censoring():
    values=trace(5)
    values[4,diagnostic.COLUMNS.index('tilt_deg')]=60
    result=diagnostic.describe_trace(values,termination='fallen_too_long',physical_cliff=True)
    assert result['edge_plus_5s_clean'] is False


def test_boundary_without_five_seconds_is_not_success():
    result=diagnostic.describe_trace(trace(5),termination='diagnostic_tile_boundary',physical_cliff=True)
    assert result['boundary_censored']
    assert result['edge_plus_5s_clean'] is None


def test_complete_followthrough_is_observed():
    result=diagnostic.describe_trace(trace(),termination='time_out',physical_cliff=True)
    assert result['edge_plus_5s_clean'] is True
    assert result['goal_plus_5s_clean'] is None


def test_cfg_preserves_success_detection_without_success_termination():
    cfg=diagnostic.diagnostic_cfg(4,0,.55,'cliff-heading',12.)
    assert cfg.episode_length_s==12
    assert cfg.terminations['cliff_landing'].func is diagnostic.record_goal_without_ending
    assert cfg.terminations['cliff_landing'].params['landing_distance']==.4
    assert cfg.rewards['cliff_landing'].weight==80
    assert cfg.observations['actor'].terms['cliff_terrain'].params['drop_height']==.02


def test_controls_change_only_declared_factor():
    zero=diagnostic.diagnostic_cfg(4,0,.55,'cliff-zero-yaw',12.)
    assert zero.commands['twist'].heading_command
    assert zero.commands['twist'].heading_control_stiffness==0.0
    assert zero.commands['twist'].ranges.ang_vel_z==(-.3,.3)
    assert zero.scene.terrain.terrain_generator.sub_terrains['cliff_drop'].min_drop==.02
    sham=diagnostic.diagnostic_cfg(4,0,.55,'sham-plane-heading',12.)
    assert sham.scene.terrain.terrain_type=='plane'
    assert sham.observations['actor'].terms['cliff_terrain'].params['drop_height']==.02
    assert sham.commands['twist'].heading_command


def test_trace_paths_do_not_overwrite_different_approaches():
    paths=[Path(diagnostic.case_label(Path('model_500.pt'),'cliff-heading',0,d)).with_suffix('.npz')
           for d in [.35,.55,.75]]
    assert len(set(paths))==3


def test_tilt_only_failure_can_be_serialized():
    import json
    values=trace()
    values[3,diagnostic.COLUMNS.index('goal_pulse')]=1
    values[5:,diagnostic.COLUMNS.index('tilt_deg')]=60
    result=diagnostic.describe_trace(values,termination='time_out',physical_cliff=True)
    assert result['original_goal_then_failure'] is True
    json.dumps(result)
