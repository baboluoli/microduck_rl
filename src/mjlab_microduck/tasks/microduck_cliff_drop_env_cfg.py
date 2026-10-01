"""Straight approach, unsupported drop, safe landing, and continued walking."""

from copy import deepcopy
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
    SERVO_GEOM_SUFFIX,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.cliff_terrain import CliffDropTerrainCfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    MicroduckRlCfg,
    _soften_terrain_contacts,
    make_microduck_velocity_env_cfg,
)


EDGE_X = 1.5
SPAWN_X = 0.95
EDGE_DISTANCE = EDGE_X - SPAWN_X
LANDING_DISTANCE = 0.40
DROP_HEIGHT_RANGE = (0.02, 0.10)
HEIGHT_BUCKETS_CM = (2, 4, 6, 8, 10)


def _max_height_bucket() -> int:
    """Training stage is advanced externally only after paired evaluation gates."""
    value = int(os.environ.get("MICRODUCK_CLIFF_MAX_HEIGHT_CM", "2"))
    if value not in HEIGHT_BUCKETS_CM:
        raise ValueError(f"MICRODUCK_CLIFF_MAX_HEIGHT_CM must be one of {HEIGHT_BUCKETS_CM}")
    return HEIGHT_BUCKETS_CM.index(value)


def make_microduck_cliff_drop_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build a fall-and-landing task with the deployed 61→14 policy interface."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)
    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    cfg.episode_length_s = 10.0

    max_bucket = len(HEIGHT_BUCKETS_CM) - 1 if play else _max_height_bucket()
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(5.0, 3.0),
            border_width=20.0,
            curriculum=True,
            num_rows=len(HEIGHT_BUCKETS_CM),
            num_cols=20 if not play else 5,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "cliff_drop": CliffDropTerrainCfg(
                    proportion=1.0,
                    edge_x=EDGE_X,
                    spawn_x=SPAWN_X,
                    min_drop=DROP_HEIGHT_RANGE[0],
                    max_drop=DROP_HEIGHT_RANGE[1],
                ),
            },
            add_lights=False,
        ),
        max_init_terrain_level=max_bucket,
    )
    cfg.scene.spec_fn = _soften_terrain_contacts
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    # Keep the inherited twist slot semantics: yaw-rate command, corrected
    # toward world +X by mjlab's proportional heading controller.
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 1.0
    twist.heading_command = True
    twist.ranges.heading = (0.0, 0.0)
    twist.heading_control_stiffness = 2.0
    twist.rel_turn_in_place_envs = 0.0
    twist.ranges.lin_vel_x = (0.18, 0.24)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (-0.6, 0.6)
    twist.init_velocity_prob = 0.0
    cfg.curriculum.pop("standing_envs", None)
    # Freeze inherited schedules at their easy starting values. Advancing the
    # height stage must not silently widen posture/CoM or smoothing demands.
    for name in ("head_pose_range", "body_pose_range", "com_range",
                 "head_com_range", "action_rate_weight", "head_pose_bias_weight"):
        cfg.curriculum.pop(name, None)
    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.rewards["head_pose_bias"].weight = 0.0
    pose = cfg.events["reset_base"].params["pose_range"]
    pose["x"] = (0.0, 0.0)
    pose["y"] = (0.0, 0.0)
    pose["yaw"] = (0.0, 0.0)
    cfg.events.pop("push_robot", None)
    for name in ("randomize_com", "randomize_head_com"):
        if name in cfg.events:
            cfg.events[name].params["ranges"] = (-0.003, 0.003)
    cfg.events["cliff_midair_reset"] = EventTermCfg(
        func=microduck_mdp.cliff_midair_reset,
        mode="reset",
        params={
            "edge_distance": EDGE_DISTANCE,
            "probability": 0.35 if not play else 0.0,
        },
    )

    # The normal walking task resets on a fall. Here falling and recovering
    # are part of the episode, and all body parts can collide with the floor.
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["fallen_too_long"] = TerminationTermCfg(
        func=microduck_mdp.fallen_too_long,
        time_out=False,
        # World z cannot classify a fall because the lower floor varies in z.
        params={"gate_z_below": -0.5, "gate_tilt_above_deg": 45.0, "max_duration_s": 4.0},
    )

    servo_ground = ContactSensorCfg(
        name="servo_ground_contact",
        primary=ContactMatch(mode="geom", pattern=rf"^.*{SERVO_GEOM_SUFFIX}$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    head_ground = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern=rf"^({'|'.join(HEAD_BODY_NAMES)})$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    trunk_ground = ContactSensorCfg(
        name="trunk_ground_contact",
        primary=ContactMatch(mode="body", pattern="^trunk_base$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    lower_contact = ContactSensorCfg(
        name="cliff_foot_contact",
        primary=ContactMatch(
            mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "pos", "force"),
        reduce="maxforce",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (servo_ground, head_ground, trunk_ground, lower_contact)
    cfg.terminations["nan_state"].params["sensor_names"] = (
        "feet_ground_contact", servo_ground.name, head_ground.name, trunk_ground.name,
        lower_contact.name,
    )

    # Keep the walk incentives, but allow a landing and charge hard impacts.
    cfg.rewards["air_time"] = RewardTermCfg(
        func=microduck_mdp.feet_air_time_upright,
        weight=cfg.rewards["air_time"].weight,
        params={**cfg.rewards["air_time"].params, "gate_tilt_above_deg": 40.0},
    )
    cfg.rewards["head_pose_bias"].params.update({
        # World height changes with the drop; use tilt to gate this term.
        "gate_height_low": -0.20,
        "gate_height_high": -0.15,
        "gate_tilt_full_deg": 20.0,
        "gate_tilt_zero_deg": 40.0,
    })
    cfg.rewards["upright_progress"] = RewardTermCfg(
        func=microduck_mdp.upright_progress,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    for name, sensor, threshold, weight in (
        ("servo_impact", servo_ground.name, 2.0, -0.02),
        ("head_impact", head_ground.name, 15.0, -0.01),
        ("trunk_impact", trunk_ground.name, 20.0, -0.01),
    ):
        cfg.rewards[name] = RewardTermCfg(
            func=microduck_mdp.body_impact_cost,
            weight=weight,
            params={"sensor_name": sensor, "threshold": threshold},
        )
    cfg.rewards["servo_acc_spike"] = RewardTermCfg(
        func=microduck_mdp.servo_acc_spike_penalty,
        weight=-1e-3,
        params={"acc_thresh": 300.0},
    )
    cfg.rewards["servo_stall"] = RewardTermCfg(
        func=microduck_mdp.servo_stall_penalty,
        weight=-0.05,
        params={"torque_thresh": 0.4, "vel_thresh": 0.5},
    )

    completion_params = {
        "edge_distance": EDGE_DISTANCE,
        "landing_distance": LANDING_DISTANCE,
    }
    cfg.rewards["cliff_landing"] = RewardTermCfg(
        func=microduck_mdp.cliff_landing_reward,
        # Success ends the episode early, forfeiting the remaining walking
        # annuity. This bonus must exceed that opportunity cost.
        weight=80.0,
        params=completion_params,
    )
    cfg.terminations["cliff_landing"] = TerminationTermCfg(
        func=microduck_mdp.cliff_landing_complete,
        time_out=False,
        params=completion_params,
    )
    cfg.curriculum.pop("terrain_levels", None)
    return cfg


MicroduckCliffDropRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckRlCfg)
MicroduckCliffDropRlCfg.experiment_name = "cliff_drop"
MicroduckCliffDropRlCfg.run_name = "cliff_drop"
MicroduckCliffDropRlCfg.max_iterations = 10_000
