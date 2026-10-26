"""Straight approach, unsupported drop, safe landing, and continued walking."""

from copy import deepcopy
import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
    SERVO_GEOM_SUFFIX,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.cliff import mdp as cliff_mdp
from mjlab_microduck.tasks.cliff.terrain import CliffDropTerrainCfg
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
    # Read the maximum allowed training height in centimetres; default to the easiest drop.
    value = int(os.environ.get("MICRODUCK_CLIFF_MAX_HEIGHT_CM", "2"))
    if value not in HEIGHT_BUCKETS_CM:  # Reject heights that do not correspond to one of the generated terrain rows.
        raise ValueError(f"MICRODUCK_CLIFF_MAX_HEIGHT_CM must be one of {HEIGHT_BUCKETS_CM}")
    return HEIGHT_BUCKETS_CM.index(value)  # Convert the height to a zero-based row index: 2 cm becomes 0, 4 cm becomes 1, etc.


def make_microduck_cliff_drop_env_cfg(
    # Build settings for policy playback; this allows starts across all height rows and uses fewer env variants.
    play: bool = False,
    *,
    profile: str | None = None,  # Select the heading or steering command setup.
    fixed_drop: float | None = None,  # Force one drop height for fixed-height experiments or evaluation.
) -> ManagerBasedRlEnvCfg:
    """Build a fall-and-landing task with the deployed 61→14 policy interface."""
    # Start with the walking task: keep its actuators, observations, actions, and randomization.
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)
    # Use collision geometry on all robot parts so head, trunk, and servo impacts are simulated.
    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    cfg.episode_length_s = 10.0  # End each episode after ten simulated seconds if it has not ended earlier.

    # Limit initial training assignments to the selected height bucket; fixed-height runs use row zero.
    max_bucket = (0 if fixed_drop is not None else
                  len(HEIGHT_BUCKETS_CM) - 1 if play else _max_height_bucket())
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",  # Build terrain from the procedural generator below.
        terrain_generator=TerrainGeneratorCfg(
            size=(5.0, 3.0),  # Make each generated terrain tile five metres along X and three metres along Y.
            border_width=20.0,  # Add a 20-metre-wide border around the terrain grid.
            curriculum=True,  # Order rows by increasing cliff height; this does not automatically advance training.
            num_rows=len(HEIGHT_BUCKETS_CM),  # Set five grid rows along X, one per drop height: 2, 4, 6, 8, and 10 cm.
            # Request grid columns along Y; mjlab uses one here because there is one terrain type and curriculum is on.
            num_cols=20 if not play else 5,
            # Set the difficulty values used to build successive grid rows; the cliff generator maps them to drop heights.
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "cliff_drop": CliffDropTerrainCfg(
                    proportion=1.0,  # Give this cliff terrain all spawn probability; it is the only terrain type configured here.
                    edge_x=EDGE_X,  # Place the cliff edge at this X coordinate in the patch.
                    spawn_x=SPAWN_X,  # Place the robot's spawn point at this X coordinate.
                    # Set the lower drop bound, or the fixed experimental height.
                    min_drop=fixed_drop if fixed_drop is not None else DROP_HEIGHT_RANGE[0],
                    max_drop=fixed_drop if fixed_drop is not None else DROP_HEIGHT_RANGE[1],  # Set the upper drop bound, or the same fixed height.
                ),
            },
            add_lights=False,  # Do not add a directional light above the terrain grid.
        ),
        max_init_terrain_level=max_bucket,  # Allow initial placement only on rows 0 through this height-stage index.
    )
    cfg.scene.spec_fn = _soften_terrain_contacts  # Apply the project's softer contact settings to generated terrain.
    cfg.sim.nconmax = 200  # Reserve capacity for up to 200 collision contact points in one simulation step.
    cfg.sim.mujoco.iterations = 30  # Repeat physics calculations up to 30 times per step to resolve motion and collisions.
    cfg.sim.mujoco.ls_iterations = 50  # Allow up to 50 extra search calculations per step to improve the physics solution.

    # Keep the inherited twist slot semantics: yaw-rate command, corrected
    # toward world +X by mjlab's proportional heading controller.
    twist = cfg.commands["twist"]
    # Expand twist so it hardcodes forward motion to walk towards the cliff.
    twist.rel_standing_envs = 0.0  # Do not assign any robots a standing-only command.
    twist.rel_heading_envs = 1.0  # Give every robot a target heading instead of an independently sampled yaw rate.
    twist.heading_command = True  # Convert heading error into a requested turning speed.
    twist.ranges.heading = (0.0, 0.0)  # Aim along world +X, which is the direction toward and beyond the cliff.
    twist.rel_turn_in_place_envs = 0.0  # Do not allocate episodes to turning on the spot.
    twist.ranges.lin_vel_x = (0.18, 0.24)  # Sample forward walking targets between 0.18 and 0.24 metres per second.
    twist.ranges.lin_vel_y = (0.0, 0.0)  # Request no sideways walking.
    twist.init_velocity_prob = 0.0  # Disable the command generator's optional initial-velocity injection.

    # Remove some existing values from the regular Tinyduck config for changing up some walking parameters with time that we don't need.

    cfg.curriculum.pop("standing_envs", None)  # Remove the inherited schedule that would change the fraction of standing commands.
    # Freeze inherited schedules at their easy starting values. Advancing the
    # height stage must not silently widen posture/CoM or smoothing demands.
    for name in ("head_pose_range", "body_pose_range", "com_range",
                 "head_com_range", "action_rate_weight", "head_pose_bias_weight"):
        cfg.curriculum.pop(name, None)
    cfg.rewards["action_rate_l2"].weight = -0.1  # Penalize abrupt changes between consecutive actions to discourage jitter.
    cfg.rewards["head_pose_bias"].weight = 0.0  # Disable the inherited head-pose bias reward for this task.
    pose = cfg.events["reset_base"].params["pose_range"]  # Edit the existing base-reset ranges; other inherited reset settings remain active.
    pose["x"] = (0.0, 0.0)  # Keep the reset position at the tile spawn rather than adding random X offsets.
    pose["y"] = (0.0, 0.0)  # Keep the robot centered across the tile.
    pose["yaw"] = (0.0, 0.0)  # Start facing +X without random yaw offsets.
    cfg.events.pop("push_robot", None)  # Remove random external pushes so the cliff is the main balance disturbance.
    for name in ("randomize_com", "randomize_head_com"):  # Keep body and head centre-of-mass uncertainty small: offsets within +/-3 mm.
        if name in cfg.events:
            cfg.events[name].params["ranges"] = (-0.003, 0.003)
    # The normal walking task resets on a fall. Here falling and recovering
    # are part of the episode, and all body parts can collide with the floor.
    cfg.terminations.pop("fell_over", None)
    # Detect servo housings hitting the floor and penalize.
    # Measure terrain forces on servo housings for the servo-impact penalty.
    servo_ground = ContactSensorCfg(
        name="servo_ground_contact",
        primary=ContactMatch(mode="geom", pattern=rf"^.*{SERVO_GEOM_SUFFIX}$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),  # Only measure contacts against terrain, excluding robot self-contact.
        fields=("force",),
        reduce="netforce",  # Combine simultaneous contact forces into a net force.
        num_slots=1,
    )
    # Detect impacts to the head and penalize.
    # Measure head impacts for both the impact penalty and landing-success checks.
    head_ground = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern=rf"^({'|'.join(HEAD_BODY_NAMES)})$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),  # Only measure contacts against terrain, excluding robot self-contact.
        fields=("force",),
        reduce="netforce",  # Combine simultaneous contact forces into a net force.
        num_slots=1,
    )
    # Detect impacts to the trunk and penalize.
    # Measure trunk impacts for both the impact penalty and landing-success checks.
    trunk_ground = ContactSensorCfg(
        name="trunk_ground_contact",
        primary=ContactMatch(mode="body", pattern="^trunk_base$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),  # Only measure contacts against terrain, excluding robot self-contact.
        fields=("force",),
        reduce="netforce",  # Combine simultaneous contact forces into a net force.
        num_slots=1,
    )
    # Record foot contact positions and forces in addition to the inherited foot sensor.
    lower_contact = ContactSensorCfg(
        name="cliff_foot_contact",
        primary=ContactMatch(
            mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),  # Only measure contacts against terrain, excluding robot self-contact.
        fields=("found", "pos", "force"),
        reduce="maxforce",  # Keep the strongest foot contact rather than summing separate contacts.
        num_slots=1,
    )
    # Append these sensors while preserving the walking task's existing sensors.
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (servo_ground, head_ground, trunk_ground, lower_contact)
    cfg.terminations["nan_state"].params["sensor_names"] = (  # Include the added sensors in the guard that resets invalid numerical states.
        "feet_ground_contact", servo_ground.name, head_ground.name, trunk_ground.name,
        lower_contact.name,
    )

    # Inherited from parent cfg.
    cfg.rewards["air_time"] = RewardTermCfg(  # Keep the walking air-time reward, but suppress it when tilt exceeds 40 degrees.
        func=microduck_mdp.feet_air_time_upright,
        weight=cfg.rewards["air_time"].weight,
        params={**cfg.rewards["air_time"].params, "gate_tilt_above_deg": 40.0},
    )
    cfg.rewards["head_pose_bias"].params.update({  # Configure the head-pose gate even though its reward weight is currently zero.
        "gate_height_low": -0.20,  # World height changes with the drop; use tilt to gate this term.
        "gate_height_high": -0.15,
        "gate_tilt_full_deg": 20.0,
        "gate_tilt_zero_deg": 40.0,
    })
    cfg.rewards["upright_progress"] = RewardTermCfg(  # Reward becoming more upright and penalize tipping further; holding a pose earns zero.
        func=microduck_mdp.upright_progress,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    for name, sensor, threshold, weight in (  # Charge for contact forces above these thresholds in newtons; negative weights make costs.
        ("servo_impact", servo_ground.name, 2.0, -0.02),
        ("head_impact", head_ground.name, 15.0, -0.01),
        ("trunk_impact", trunk_ground.name, 20.0, -0.01),
    ):
        cfg.rewards[name] = RewardTermCfg(
            func=microduck_mdp.body_impact_cost,
            weight=weight,
            params={"sensor_name": sensor, "threshold": threshold},
        )
    cfg.rewards["servo_acc_spike"] = RewardTermCfg(  # Penalize joint acceleration above 300 rad/s² to discourage harsh landing loads.
        func=microduck_mdp.servo_acc_spike_penalty,
        weight=-1e-3,
        params={"acc_thresh": 300.0},
    )
    cfg.rewards["servo_stall"] = RewardTermCfg(  # Penalize servos exerting over 0.4 N·m while moving slower than 0.5 rad/s.
        func=microduck_mdp.servo_stall_penalty,
        weight=-0.05,
        params={"torque_thresh": 0.4, "vel_thresh": 0.5},
    )

    completion_params = {  # Share the same edge and post-edge travel distances between success reward and termination.
        "edge_distance": EDGE_DISTANCE,
        "landing_distance": LANDING_DISTANCE,  # Require travel at least 0.40 metres beyond the edge before success is possible.
    }
    cfg.rewards["cliff_landing"] = RewardTermCfg(  # Pay a one-time bonus for a stable, feet-supported landing with continued forward motion.
        func=cliff_mdp.cliff_landing_reward,
        # Success ends the episode early, forfeiting the remaining walking
        # annuity. This bonus must exceed that opportunity cost.
        weight=80.0,
        params=completion_params,
    )
    # Success requires 0.30 s of supported forward travel with tilt <=30 degrees,
    # limited vertical speed, and no disqualifying head or trunk impact.
    # End successful episodes using exactly the same condition as the landing bonus.
    cfg.terminations["cliff_landing"] = TerminationTermCfg(
        func=cliff_mdp.cliff_landing_complete,
        time_out=False,  # Treat this as a terminal outcome rather than a time-limit truncation.
        params=completion_params,
    )
    cfg.curriculum.pop("terrain_levels", None)  # Disable automatic movement between height rows; the maximum training height is set externally.
    # Prefer an explicit profile argument; otherwise read the environment variable, defaulting to heading.
    profile = profile if profile is not None else os.environ.get("MICRODUCK_CLIFF_PROFILE", "heading")
    # Each branch owns all profile-specific settings; shared commands above apply to both.
    if profile == "heading":
        twist.heading_control_stiffness = 2.0  # Correct heading errors more strongly.
        twist.ranges.ang_vel_z = (-0.6, 0.6)  # Allow faster turning corrections, in rad/s.
        midair_probability = 0.0 if play else 0.35  # Practice landings directly during training.
        recovery_duration_s = 4.0  # Allow time to recover after a fall.
        # Keep the velocity-tracking reward inherited from the walking task.
    elif profile == "steering":
        # Requires prepare_cliff_steering_checkpoint.py for a zero-yaw
        # source: its unused input column must not receive arbitrary new
        # normalized values when the heading controller is enabled.
        twist.heading_control_stiffness = 0.5  # Use gentler heading correction than the default heading profile.
        twist.ranges.ang_vel_z = (-0.3, 0.3)  # Restrict steering corrections to +/-0.3 radians per second.
        # Train the full approach and descent instead of starting partway through the landing.
        midair_probability = 0.0
        # Replace velocity tracking with capped world-X progress, so standing still earns no forward credit.
        cfg.rewards["track_linear_velocity"] = RewardTermCfg(
            func=cliff_mdp.cliff_forward_progress, weight=4.0,
        )
        # End attempts immediately when head or trunk impacts disqualify a successful landing.
        cfg.terminations["cliff_failed_landing"] = TerminationTermCfg(
            func=cliff_mdp.cliff_failed_landing,
            params=dict(completion_params), time_out=False,
        )
        # Shorten the recovery window to half a second for the steering profile.
        recovery_duration_s = 0.5
    else:  # Reject unrecognized profile names.
        raise ValueError("MICRODUCK_CLIFF_PROFILE must be heading or steering")

    # Wire the reset and fall timer once, using the selected profile values.
    cfg.events["cliff_midair_reset"] = EventTermCfg(  # Start some training episodes already in flight so landing receives experience early.
        func=cliff_mdp.cliff_midair_reset,
        mode="reset",
        params={
            "edge_distance": EDGE_DISTANCE,  # Locate the edge 0.55 metres ahead of the normal spawn.
            "probability": midair_probability,  # Use the reset mix selected by the profile.
        },
    )

    cfg.terminations["fallen_too_long"] = TerminationTermCfg(
        func=microduck_mdp.fallen_too_long,
        time_out=False,  # Treat this as a terminal outcome rather than a time-limit truncation.
        # World z cannot classify a fall because the lower floor varies in z.
        params={"gate_z_below": -0.5, "gate_tilt_above_deg": 45.0, "max_duration_s": recovery_duration_s},
    )

    return cfg  # Return the configuration; simulation and training are started elsewhere.


MicroduckCliffDropRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckRlCfg)
MicroduckCliffDropRlCfg.experiment_name = "cliff_drop"
MicroduckCliffDropRlCfg.run_name = "cliff_drop"
MicroduckCliffDropRlCfg.max_iterations = 10_000


ORACLE_DROP_HEIGHT = 0.02  # Experimental oracle interface: deliberately separate from deployed 61D policies.


def make_microduck_cliff_oracle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Fixed 2 cm, upper-start descent with exact terrain inputs (63→14).

    Appends [signed world-X edge distance, positive drop height], in metres,
    to both observation groups. This is a simulation-only interface; it needs
    a matching terrain estimator before any hardware deployment.
    """
    # Choose steering for its full-approach starts, progress reward, and strict failures.
    # Exact terrain observations below are independent of that choice of training profile.
    cfg = make_microduck_cliff_drop_env_cfg(
        play=play, profile="steering", fixed_drop=ORACLE_DROP_HEIGHT,
    )
    # Use the full requested grid with random allocation; all tiles still have the same fixed height.
    cfg.scene.terrain.terrain_generator.curriculum = False
    for group in ("actor", "critic"):  # Give both the action-producing policy (actor) and value estimator (critic) the two extra inputs.
        # Append exact signed edge distance and drop height, increasing observations from 61 to 63 values.
        cfg.observations[group].terms["cliff_terrain"] = ObservationTermCfg(
            func=cliff_mdp.cliff_terrain_observation,
            params={"edge_distance": EDGE_DISTANCE, "drop_height": ORACLE_DROP_HEIGHT},
        )
    cfg.terminations.pop("out_of_terrain_bounds", None)  # Remove the inherited terrain-boundary termination for this oracle experiment.
    return cfg  # Return the configuration; simulation and training are started elsewhere.


MicroduckCliffOracleRlCfg = deepcopy(MicroduckCliffDropRlCfg)
MicroduckCliffOracleRlCfg.experiment_name = "cliff_oracle_2cm"
MicroduckCliffOracleRlCfg.run_name = "cliff_oracle_2cm"

MicroduckCliffOracleRlCfg.algorithm.symmetry_cfg = None  # The deployed symmetry map covers 61 inputs, so disable it for this interface.


def make_microduck_cliff_walkthrough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Continue walking for the full horizon, always starting 55 cm from edge."""
    from . import walkthrough
    cfg = make_microduck_cliff_oracle_env_cfg(play=play)
    cfg.episode_length_s = 12.0
    cfg.terminations.pop('cliff_landing')
    cfg.terminations['walkthrough_boundary'] = TerminationTermCfg(
        func=walkthrough.boundary, time_out=True)
    cfg.rewards['cliff_landing'] = RewardTermCfg(func=walkthrough.reward, weight=1.)
    return cfg


MicroduckCliffWalkThroughRlCfg = deepcopy(MicroduckCliffOracleRlCfg)
MicroduckCliffWalkThroughRlCfg.experiment_name = 'oracle_walkthrough_fixed'
MicroduckCliffWalkThroughRlCfg.run_name = 'oracle_walkthrough_fixed'


def make_microduck_cliff_course_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Walk 1 m past landing, immutable height bank (2–3 cm at 0.5 mm, 3.25–10 cm at 2.5 mm, 10.5–14 cm at 5 mm), randomized approach only."""
    from . import course
    cfg = make_microduck_cliff_oracle_env_cfg(play)
    cfg.episode_length_s = 12.
    generator = cfg.scene.terrain.terrain_generator
    generator.curriculum = True
    generator.num_rows = len(course.HEIGHTS)
    generator.difficulty_range = (0., 1.)
    tile = generator.sub_terrains['cliff_drop']
    tile.min_drop, tile.max_drop = course.HEIGHTS[0], course.HEIGHTS[-1]
    tile.height_buckets = course.HEIGHTS
    tile.center_in_tile_y = True  # Keep the complete course clear of the flat grid border.
    cfg.scene.terrain.max_init_terrain_level = 0
    for group in ('actor', 'critic'):
        cfg.observations[group].terms['cliff_terrain'] = ObservationTermCfg(func=course.observation)
    cfg.events['course_reset'] = EventTermCfg(func=course.reset, mode='reset',
        params={'upper': .02, 'start_range': (.55, .55) if play else (.45, .65)})
    cfg.terminations.pop('cliff_landing')
    cfg.terminations['course_success'] = TerminationTermCfg(func=course.success)
    # Leaving the permitted course is a task failure. A timeout would tell PPO
    # to add the estimated remaining value back onto this failed transition.
    cfg.terminations['course_boundary'] = TerminationTermCfg(func=course.boundary)
    cfg.rewards['cliff_landing'] = RewardTermCfg(func=course.reward, weight=1.)
    # Progress counts world X only, so nothing else prices sideways drift.
    # Weight 5 charges model_99's 0.13 m/s crab ~0.65/s, about 16% of the
    # 4/s full-speed progress reward; the source's 0.056 m/s costs ~0.28/s.
    cfg.rewards['sideways_drift'] = RewardTermCfg(func=course.sideways_drift, weight=5.)
    return cfg


MicroduckCliffCourseRlCfg = deepcopy(MicroduckCliffOracleRlCfg)
MicroduckCliffCourseRlCfg.experiment_name = 'oracle_course'
MicroduckCliffCourseRlCfg.run_name = 'oracle_course'
# The entropy bonus rewards wider exploration noise. On the course it inflated
# the action std every 100-update block (0.96 -> 1.43 over the 2026-10-25
# height session) while walking degraded. Weaker bonus, course task only.
MicroduckCliffCourseRlCfg.algorithm.entropy_coef = 0.003


def make_microduck_cliff_course_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Course terrain and starts, but trained as plain walking: no finish, no completion credit.

    Why (hold-height session, 2026-10-25): standing upright earns ~3 points/s
    (upright, head_pose_tracking, pose, air_time). When success ENDS the episode,
    that income stops, so the last few centimetres before the goal are worth
    less than standing still. With 80 points/m and gamma 0.99 at 50 Hz the
    break-even is ~8 cm, and the policy loitered 6–8 cm short of the 1 m goal.
    Paying distance (not walking) also made rushing free: 0.6 m/s against a
    0.18–0.3 m/s command.

    So here the cliff is just terrain on a walk:
    - no success termination: every attempt runs to the (bootstrapped) timeout,
      so there is no end worth avoiding;
    - no completion credit: the 80-point course budget is removed;
    - the walking task's velocity tracking is back. Its exp kernel peaks at the
      commanded speed and charges overspeed and sideways motion, which the
      capped progress term never did.

    Success (1 m clean walking after touchdown) stays an evaluation metric:
    ``scripts/eval_cliff_course.py`` still uses the 12 s course cfg above.
    Same 63/78/14 interface, so checkpoints move freely between the two tasks.
    """
    cfg = make_microduck_cliff_course_env_cfg(play)
    # Episodes no longer stop at the goal, so 12 s is mostly flat walking after
    # the landing. 8 s still leaves ~5 s after touchdown from the farthest 65 cm
    # start at the commanded ~0.21 m/s, and gives ~1.5x more landings per step.
    cfg.episode_length_s = 8.
    cfg.terminations.pop('course_success')
    cfg.rewards.pop('cliff_landing')
    # The steering profile renamed capped world-X progress to this slot. Keep it,
    # at half weight, so standing at the edge still pays less than stepping off;
    # it pays nothing above the command, and the tracker below charges for that.
    progress = cfg.rewards.pop('track_linear_velocity')
    progress.weight = 2.
    cfg.rewards['forward_progress'] = progress
    # Same term and settings as microduck_velocity_env_cfg (weight 2, std^2 0.1):
    # at a 0.2 m/s command, 0.6 m/s earns exp(-0.16/0.1) = 20% of the peak.
    cfg.rewards['track_linear_velocity'] = RewardTermCfg(
        func=velocity_mdp.track_linear_velocity,
        weight=2.,
        params={'command_name': 'twist', 'std': math.sqrt(0.1)},
    )
    # This task is trained FROM SCRATCH, so give it the walking recipe's
    # schedules back. make_microduck_cliff_drop_env_cfg froze them at their
    # stage-0 values for warm-started height stages; that left smoothing
    # (action_rate_l2) at -0.1 for every cliff run, 10x weaker than the -1.0
    # the source walking policy was trained under. ``standing_envs`` stays off:
    # the course commands forward walking only.
    walking = make_microduck_velocity_env_cfg(play=play, rough=False)
    for name in COURSE_WALK_CURRICULA:
        cfg.curriculum[name] = deepcopy(walking.curriculum[name])
    for name in ('action_rate_l2', 'head_pose_bias'):  # Stage-0 weights the schedules ramp from.
        cfg.rewards[name].weight = walking.rewards[name].weight
    for name in ('randomize_com', 'randomize_head_com'):  # The CoM schedules widen these.
        cfg.events[name].params['ranges'] = walking.events[name].params['ranges']
    return cfg


COURSE_WALK_CURRICULA = ('action_rate_weight', 'com_range', 'head_com_range',
                         'head_pose_range', 'body_pose_range', 'head_pose_bias_weight')


MicroduckCliffCourseWalkRlCfg = deepcopy(MicroduckCliffCourseRlCfg)
MicroduckCliffCourseWalkRlCfg.experiment_name = 'oracle_course_walk'
MicroduckCliffCourseWalkRlCfg.run_name = 'oracle_course_walk'
# The course task's 0.003 curbed action-std creep while FINE-TUNING. From
# scratch the gait must first be discovered: use the walking recipe's 0.01.
MicroduckCliffCourseWalkRlCfg.algorithm.entropy_coef = MicroduckRlCfg.algorithm.entropy_coef
MicroduckCliffCourseWalkRlCfg.max_iterations = 3000
