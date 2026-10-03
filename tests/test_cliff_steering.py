"""Regression cases for steering transfer and direction-aware reward."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp

spec = importlib.util.spec_from_file_location(
    "prepare_steering", Path(__file__).parents[1] / "scripts/prepare_cliff_steering_checkpoint.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_new_yaw_input_preserves_original_network_response():
    torch.manual_seed(42)
    source = {"optimizer_state_dict": {"state": {0: {"step": 12}}, "param_groups": [{"lr": 0.001}]},
              "infos": {"env_state": {"common_step_counter": 24000}}, "iter": 999}
    for key, width, yaw in (("actor_state_dict", 61, 50), ("critic_state_dict", 76, 53)):
        state = {"mlp.0.weight": torch.randn(32, width),
                 "mlp.0.bias": torch.randn(32),
                 "obs_normalizer._mean": torch.randn(1, width),
                 "obs_normalizer._std": torch.rand(1, width) + 0.1}
        state["obs_normalizer._mean"][0, yaw] = 0
        state["obs_normalizer._std"][0, yaw] = 0
        state["obs_normalizer._var"] = state["obs_normalizer._std"].square()
        source[key] = state
    result = prepare.prepare_checkpoint(source)
    for key, width, yaw in (("actor_state_dict", 61, 50), ("critic_state_dict", 76, 53)):
        old, new = source[key], result[key]
        obs = torch.randn(101, width)
        obs[:, yaw] = 0
        def response(s, x):
            normalized = (x - s["obs_normalizer._mean"]) / (s["obs_normalizer._std"] + 0.01)
            return normalized @ s["mlp.0.weight"].T + s["mlp.0.bias"]
        expected = response(old, obs)
        obs[:, yaw] = torch.linspace(-0.3, 0.3, 101)
        torch.testing.assert_close(response(new, obs), expected, rtol=0, atol=0)
        assert old["mlp.0.weight"][:, yaw].abs().sum() > 0  # source untouched
        assert new["obs_normalizer._std"][0, yaw] > 0
    assert result["optimizer_state_dict"]["state"] == {}
    assert result["optimizer_state_dict"]["param_groups"][0]["lr"] == 1e-5
    assert result["infos"]["env_state"]["common_step_counter"] == 0
    assert source["iter"] == 999
    with pytest.raises(ValueError, match="never-used"):
        prepare.prepare_checkpoint(result)


def test_progress_reward_does_not_pay_standing_sideways_or_fallen():
    # Forward, standing, sideways, backwards, too-fast forward, fallen forward.
    velocity = torch.tensor([[.2, 0, 0], [0, 0, 0], [0, .2, 0], [-.2, 0, 0],
                             [.8, 0, 0], [.2, 0, 0]])
    quat = torch.tensor([[1., 0, 0, 0]] * 5 + [[0., 1, 0, 0]])
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_w=velocity,
                                                              root_link_quat_w=quat))},
        command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[.2, 0, 0]] * 6)),
    )
    torch.testing.assert_close(mdp.cliff_forward_progress(env), torch.tensor([1., 0, 0, -1, 1, 0]))


def test_failed_landing_uses_the_success_disqualification(monkeypatch):
    env = SimpleNamespace()
    def complete(e, **kwargs):
        e._cliff_bad_impact = torch.tensor([False, True, False])
    monkeypatch.setattr(mdp, "cliff_landing_complete", complete)
    assert mdp.cliff_failed_landing(env).tolist() == [False, True, False]
