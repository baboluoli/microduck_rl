"""A genuine ledge with a lower landing floor, sampled at several heights."""

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class CliffDropTerrainCfg(SubTerrainCfg):
    """Two flat platforms separated by an unsupported vertical drop.

    Curriculum rows are explicit height buckets. The training stage controls
    which rows can be sampled; failed episodes never lower that stage.
    """

    edge_x: float = 1.5
    spawn_x: float = 0.95
    min_drop: float = 0.02
    max_drop: float = 0.10
    height_buckets: tuple[float, ...] = (0.02, 0.04, 0.06, 0.08, 0.10)
    thickness: float = 0.3

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng) -> TerrainOutput:
        if not 0.0 < self.spawn_x < self.edge_x < self.size[0]:
            raise ValueError("cliff spawn must be before the edge and landing floor")
        if not 0.0 < self.min_drop <= self.max_drop < self.thickness:
            raise ValueError("cliff drop heights must fit inside the platform thickness")

        if self.min_drop == self.max_drop:
            drop = self.min_drop  # fixed-height evaluation
        else:
            bucket = min(int(np.clip(difficulty, 0.0, 1.0) * len(self.height_buckets)),
                         len(self.height_buckets) - 1)
            drop = self.height_buckets[bucket]
        width = self.size[1] - 0.3
        terrain = spec.body("terrain")
        geoms = []
        for x0, x1, z, color in (
            (0.0, self.edge_x, 0.0, (0.52, 0.53, 0.55, 1.0)),
            (self.edge_x, self.size[0], -drop, (0.34, 0.42, 0.46, 1.0)),
        ):
            geom = terrain.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=((x1 - x0) / 2.0, width / 2.0, self.thickness / 2.0),
                pos=((x0 + x1) / 2.0, 0.0, z - self.thickness / 2.0),
            )
            geoms.append(TerrainGeometry(geom=geom, color=color))

        return TerrainOutput(
            origin=np.array([self.spawn_x, 0.0, 0.0]),
            geometries=geoms,
        )
