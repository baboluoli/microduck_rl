"""Microduck walking task for procedurally varied approaches into a trench.

The actor remains the normal 61-observation / 14-joint policy.  PPO learns the
joint targets that follow a forward walking command over sampled flat and
sloping approaches and down/up the trench ramps; no gait or pose timeline is
scripted in the environment.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    _soften_terrain_contacts,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.hole_terrain import HoleDescentTerrainCfg


TERRAIN_SIZE = (9.0, 4.0)


def make_microduck_hole_descent_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the terrain-randomized biped descent task from the walking recipe."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)
    terrain_generator = TerrainGeneratorCfg(
        size=TERRAIN_SIZE,
        border_width=20.0,
        curriculum=not play,
        num_rows=10 if not play else 5,
        num_cols=20 if not play else 5,
        difficulty_range=(0.0, 1.0),
        sub_terrains={
            "hole_descent": HoleDescentTerrainCfg(
                proportion=1.0,
                flat_length_range=(1.2, 1.8),
                approach_length_range=(0.6, 1.2),
                approach_grade_max=0.10,
                pit_depth_range=(0.06, 0.14),
                entry_length_range=(0.65, 1.0),
                bottom_length_range=(0.6, 1.0),
                runout_length=1.5,
            ),
        },
        add_lights=False,
    )
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=terrain_generator,
        max_init_terrain_level=0 if not play else None,
    )

    # The task is one continuous forward walk. The policy still decides its
    # joint actions; the command only specifies the intended travel direction.
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0
    twist.ranges.lin_vel_x = (0.18, 0.24)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)

    # Align the initial pose with the sampled downhill course and keep the
    # spawn on the flat start pad. Other reset randomization and actuator DR
    # remain inherited from the established walking recipe.
    reset_pose = cfg.events["reset_base"].params["pose_range"]
    reset_pose["x"] = (0.0, 0.0)
    reset_pose["y"] = (0.0, 0.0)
    reset_pose["yaw"] = (0.0, 0.0)

    # Adjacent ramp/platform contacts can generate large solver impulses.
    cfg.scene.spec_fn = _soften_terrain_contacts
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    # Let successful forward travel promote the environment to harder
    # geometry rows; low-progress resets move back toward shallow layouts.
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=microduck_mdp.terrain_levels_slope,
    )

    return cfg


MicroduckHoleDescentRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckRlCfg)
MicroduckHoleDescentRlCfg.experiment_name = "hole_descent"
MicroduckHoleDescentRlCfg.run_name = "hole_descent"
MicroduckHoleDescentRlCfg.max_iterations = 10_000
