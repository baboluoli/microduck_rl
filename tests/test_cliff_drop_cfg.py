"""Cliff stage and evaluator contracts that do not require a physics rollout."""

import pytest

from mjlab_microduck.tasks.cliff_terrain import CliffDropTerrainCfg
from mjlab_microduck.tasks.microduck_cliff_drop_env_cfg import (
    HEIGHT_BUCKETS_CM,
    make_microduck_cliff_drop_env_cfg,
)


def test_cliff_heading_and_frozen_schedules(monkeypatch):
    monkeypatch.setenv("MICRODUCK_CLIFF_MAX_HEIGHT_CM", "6")
    cfg = make_microduck_cliff_drop_env_cfg()
    twist = cfg.commands["twist"]
    assert twist.heading_command
    assert twist.rel_heading_envs == 1.0
    assert twist.ranges.heading == (0.0, 0.0)
    assert twist.ranges.ang_vel_z[0] < 0 < twist.ranges.ang_vel_z[1]
    assert cfg.scene.terrain.max_init_terrain_level == 2
    assert cfg.scene.terrain.terrain_generator.num_rows == len(HEIGHT_BUCKETS_CM)
    assert cfg.curriculum == {}
    assert cfg.rewards["action_rate_l2"].weight == -0.1
    assert cfg.rewards["head_pose_bias"].weight == 0.0
    assert cfg.events["randomize_com"].params["ranges"] == (-0.003, 0.003)
    assert cfg.events["randomize_head_com"].params["ranges"] == (-0.003, 0.003)
    sensor = next(s for s in cfg.scene.sensors if s.name == "cliff_foot_contact")
    assert {"found", "pos"}.issubset(sensor.fields)


def test_cliff_height_stage_rejects_unmeasured_values(monkeypatch):
    monkeypatch.setenv("MICRODUCK_CLIFF_MAX_HEIGHT_CM", "3")
    with pytest.raises(ValueError, match="MICRODUCK_CLIFF_MAX_HEIGHT_CM"):
        make_microduck_cliff_drop_env_cfg()


def test_bucket_definition_matches_training_rows():
    terrain = CliffDropTerrainCfg()
    assert terrain.height_buckets == tuple(x / 100 for x in HEIGHT_BUCKETS_CM)


def test_model999_profile_preserves_pre_boundary_conditions(monkeypatch):
    monkeypatch.setenv("MICRODUCK_CLIFF_PROFILE", "model999")
    monkeypatch.setenv("MICRODUCK_CLIFF_MAX_HEIGHT_CM", "2")
    cfg = make_microduck_cliff_drop_env_cfg()
    twist = cfg.commands["twist"]
    assert not twist.heading_command
    assert twist.rel_heading_envs == 0.0
    assert twist.ranges.ang_vel_z == (0.0, 0.0)
    assert cfg.rewards["action_rate_l2"].weight == -0.4
    assert cfg.rewards["head_pose_bias"].weight == 1.0
    assert cfg.commands["head_pose"].ranges == (
        (-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047)
    )
    assert cfg.events["randomize_com"].params["ranges"] == (-0.005, 0.005)
    assert cfg.events["randomize_head_com"].params["ranges"] == (-0.005, 0.005)
    assert cfg.curriculum == {}
    assert cfg.scene.terrain.max_init_terrain_level == 0


def test_invalid_cliff_profile(monkeypatch):
    monkeypatch.setenv("MICRODUCK_CLIFF_PROFILE", "unknown")
    with pytest.raises(ValueError, match="MICRODUCK_CLIFF_PROFILE"):
        make_microduck_cliff_drop_env_cfg()


def test_steering_profile_keeps_interface_and_ends_disqualified_attempts(monkeypatch):
    from mjlab_microduck.tasks import mdp

    monkeypatch.setenv("MICRODUCK_CLIFF_PROFILE", "steering")
    cfg = make_microduck_cliff_drop_env_cfg()
    assert cfg.commands["twist"].heading_command
    assert cfg.commands["twist"].ranges.ang_vel_z == (-0.3, 0.3)
    assert cfg.events["cliff_midair_reset"].params["probability"] == 0
    assert cfg.rewards["track_linear_velocity"].func is mdp.cliff_forward_progress
    assert cfg.terminations["cliff_failed_landing"].func is mdp.cliff_failed_landing
    assert cfg.terminations["fallen_too_long"].params["max_duration_s"] == 0.5
    # These orders establish yaw at actor[50] and critic[53].
    base = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command"]
    assert list(cfg.observations["actor"].terms)[:6] == base
    assert list(cfg.observations["critic"].terms)[:7] == ["base_lin_vel", *base]
    assert cfg.curriculum == {}
