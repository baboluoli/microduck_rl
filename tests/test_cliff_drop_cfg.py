"""Cliff stage and evaluator contracts that do not require a physics rollout."""

import pytest

from mjlab_microduck.tasks.cliff.terrain import CliffDropTerrainCfg
from mjlab_microduck.tasks.cliff.env_cfg import (
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


@pytest.mark.parametrize("profile", ["unknown", "model999"])
def test_invalid_cliff_profile(monkeypatch, profile):
    monkeypatch.setenv("MICRODUCK_CLIFF_PROFILE", profile)
    with pytest.raises(ValueError, match="MICRODUCK_CLIFF_PROFILE"):
        make_microduck_cliff_drop_env_cfg()


def test_steering_profile_keeps_interface_and_ends_disqualified_attempts(monkeypatch):
    from mjlab_microduck.tasks.cliff import mdp

    monkeypatch.setenv("MICRODUCK_CLIFF_PROFILE", "steering")
    cfg = make_microduck_cliff_drop_env_cfg()
    assert cfg.commands["twist"].heading_command
    assert cfg.commands["twist"].ranges.ang_vel_z == (-0.3, 0.3)
    assert cfg.rewards["action_rate_l2"].weight == -0.1
    assert cfg.rewards["head_pose_bias"].weight == 0.0
    assert cfg.events["randomize_com"].params["ranges"] == (-0.003, 0.003)
    assert cfg.events["randomize_head_com"].params["ranges"] == (-0.003, 0.003)
    assert cfg.events["cliff_midair_reset"].params["probability"] == 0
    assert cfg.rewards["track_linear_velocity"].func is mdp.cliff_forward_progress
    assert cfg.terminations["cliff_failed_landing"].func is mdp.cliff_failed_landing
    assert cfg.terminations["fallen_too_long"].params["max_duration_s"] == 0.5
    # These orders establish yaw at actor[50] and critic[53].
    base = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command"]
    assert list(cfg.observations["actor"].terms)[:6] == base
    assert list(cfg.observations["critic"].terms)[:7] == ["base_lin_vel", *base]
    assert cfg.curriculum == {}


@pytest.mark.parametrize("play", [False, True])
@pytest.mark.parametrize("profile", ["heading", "steering"])
def test_profile_command_reset_and_recovery_settings(profile, play):
    cfg = make_microduck_cliff_drop_env_cfg(play=play, profile=profile)
    twist = cfg.commands["twist"]
    assert twist.rel_standing_envs == 0.0
    assert twist.rel_heading_envs == 1.0
    assert twist.heading_command
    assert twist.ranges.heading == (0.0, 0.0)
    assert twist.ranges.lin_vel_x == (0.18, 0.24)
    assert twist.ranges.lin_vel_y == (0.0, 0.0)
    assert twist.heading_control_stiffness == (2.0 if profile == "heading" else 0.5)
    assert twist.ranges.ang_vel_z == ((-0.6, 0.6) if profile == "heading" else (-0.3, 0.3))
    expected_midair = 0.35 if profile == "heading" and not play else 0.0
    assert cfg.events["cliff_midair_reset"].params["probability"] == expected_midair
    assert cfg.terminations["fallen_too_long"].params["max_duration_s"] == (
        4.0 if profile == "heading" else 0.5
    )
    assert ("cliff_failed_landing" in cfg.terminations) == (profile == "steering")
