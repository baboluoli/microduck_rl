"""Procedurally generated flat/slope approaches into a shallow trench.

Each terrain tile samples its approach length, approach grade, trench depth,
entry/exit lengths, and bottom length from the supplied RNG.  The terrain is a
set of adjoining collision boxes, matching the existing Microduck slope-task
approach and avoiding a train/render heightfield mismatch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)


@dataclass(kw_only=True)
class HoleDescentTerrainCfg(SubTerrainCfg):
    """Flat or gently sloped ground leading down into a shallow trench.

    Difficulty controls the maximum approach grade and trench depth.  Lengths
    are still sampled independently for every generated tile, so a policy sees
    different layouts even at one curriculum level.
    """

    flat_length_range: tuple[float, float] = (1.2, 1.8)
    approach_length_range: tuple[float, float] = (0.6, 1.2)
    approach_grade_max: float = 0.10
    pit_depth_range: tuple[float, float] = (0.06, 0.14)
    entry_length_range: tuple[float, float] = (0.65, 1.0)
    bottom_length_range: tuple[float, float] = (0.6, 1.0)
    runout_length: float = 1.5
    thickness: float = 0.3
    spawn_fraction: float = 0.5

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng) -> TerrainOutput:
        d = float(np.clip(difficulty, 0.0, 1.0))
        body = spec.body("terrain")
        width = self.size[1] - 0.3
        thickness = self.thickness
        geoms = []

        def add_flat(x0: float, x1: float, z: float, color=(0.52, 0.53, 0.55, 1.0)) -> None:
            if x1 - x0 <= 1e-5:
                return
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=((x1 - x0) / 2.0, width / 2.0, thickness / 2.0),
                pos=((x0 + x1) / 2.0, 0.0, z - thickness / 2.0),
            )
            geoms.append(TerrainGeometry(geom=geom, color=color))

        def add_ramp(
            x0: float,
            z0: float,
            x1: float,
            z1: float,
            color=(0.43, 0.54, 0.70, 1.0),
        ) -> None:
            dx = x1 - x0
            dz = z1 - z0
            if dx <= 1e-5:
                return
            angle = math.atan2(-dz, dx)  # positive angle descends toward +x
            surface_length = math.hypot(dx, dz)
            # Place the box centre half a thickness below its top surface.
            cx = (x0 + x1) / 2.0 - (thickness / 2.0) * math.sin(angle)
            cz = (z0 + z1) / 2.0 - (thickness / 2.0) * math.cos(angle)
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(surface_length / 2.0, width / 2.0, thickness / 2.0),
                pos=(cx, 0.0, cz),
                quat=(math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0),
            )
            geoms.append(TerrainGeometry(geom=geom, color=color))

        flat_length = float(rng.uniform(*self.flat_length_range))
        approach_length = float(rng.uniform(*self.approach_length_range))
        # Zero is a real option: some episodes have a completely flat approach.
        approach_grade = float(rng.uniform(0.0, self.approach_grade_max * d))
        approach_drop = approach_grade * approach_length
        depth_hi = self.pit_depth_range[0] + d * (self.pit_depth_range[1] - self.pit_depth_range[0])
        depth = float(rng.uniform(self.pit_depth_range[0], max(self.pit_depth_range[0], depth_hi)))
        entry_length = float(rng.uniform(*self.entry_length_range))
        bottom_length = float(rng.uniform(*self.bottom_length_range))

        x = 0.0
        z_rim = 0.0
        add_flat(x, flat_length, z_rim)
        x = flat_length

        approach_end = x + approach_length
        z_rim -= approach_drop
        if approach_drop > 1e-4:
            add_ramp(x, 0.0, approach_end, z_rim)
        else:
            add_flat(x, approach_end, z_rim)
        x = approach_end

        z_bottom = z_rim - depth
        entry_end = x + entry_length
        add_ramp(x, z_rim, entry_end, z_bottom)
        x = entry_end

        bottom_end = x + bottom_length
        add_flat(x, bottom_end, z_bottom, color=(0.34, 0.38, 0.43, 1.0))
        x = bottom_end

        exit_end = x + entry_length
        add_ramp(x, z_bottom, exit_end, z_rim)
        x = exit_end

        tile_end = self.size[0]
        assert x + self.runout_length <= tile_end, (
            f"sampled course needs {x + self.runout_length:.2f} m, "
            f"but terrain size[0] is {tile_end:.2f} m"
        )
        add_flat(x, tile_end, z_rim)

        spawn_x = flat_length * float(np.clip(self.spawn_fraction, 0.1, 0.9))
        return TerrainOutput(
            origin=np.array([spawn_x, 0.0, 0.0]),
            geometries=geoms,
        )
