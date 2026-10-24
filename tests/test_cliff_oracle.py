"""Oracle geometry, task isolation and evaluation consistency."""
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks.cliff import mdp
from mjlab_microduck.tasks.cliff.env_cfg import (
    EDGE_DISTANCE, make_microduck_cliff_drop_env_cfg, make_microduck_cliff_oracle_env_cfg,
)


def test_oracle_geometry_tracks_motion_and_tile_origin():
    # Same local position on different tiles, at edge, and past edge.
    origins = torch.tensor([[0., 0, 0], [10., 3, 0], [20., 0, 0], [30., 0, 0]])
    positions = origins + torch.tensor([[.1, 0, .2], [.1, 0, .1],
                                        [EDGE_DISTANCE, 0, .2], [.8, 0, -.1]])
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=positions))})
    class Scene(dict):
        env_origins = origins
    env.scene = Scene(env.scene)
    obs = mdp.cliff_terrain_observation(env, EDGE_DISTANCE, .02)
    torch.testing.assert_close(obs[:, 0], torch.tensor([.45, .45, 0., -.25]), atol=2e-6, rtol=0)
    torch.testing.assert_close(obs[:, 1], torch.full((4,), .02))


@pytest.mark.parametrize("play", [False, True])
def test_oracle_fixed_geometry_and_separate_interface(monkeypatch, play):
    monkeypatch.setenv("MICRODUCK_CLIFF_PROFILE", "heading")
    monkeypatch.setenv("MICRODUCK_CLIFF_MAX_HEIGHT_CM", "10")
    cfg = make_microduck_cliff_oracle_env_cfg(play=play)
    base = make_microduck_cliff_drop_env_cfg(play=play)
    for group in ("actor", "critic"):
        assert list(cfg.observations[group].terms) == list(base.observations[group].terms) + ["cliff_terrain"]
        term = cfg.observations[group].terms["cliff_terrain"]
        assert term.params == {"edge_distance": EDGE_DISTANCE, "drop_height": .02}
        assert term.noise is None
        assert "cliff_terrain" not in base.observations[group].terms
    terrain = cfg.scene.terrain.terrain_generator
    assert not terrain.curriculum
    assert cfg.scene.terrain.max_init_terrain_level == 0
    assert terrain.sub_terrains["cliff_drop"].min_drop == .02
    assert terrain.sub_terrains["cliff_drop"].max_drop == .02
    assert cfg.events["cliff_midair_reset"].params["probability"] == 0
    assert cfg.commands["twist"].heading_command
    assert cfg.curriculum == {}
