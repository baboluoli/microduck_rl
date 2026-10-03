"""Activate the unused yaw input without changing the source policy's gait.

Only for this repository's 61D actor / 76D critic zero-yaw cliff checkpoints.
Keeps every other weight and normalizer statistic, resets Adam, and restarts
counters. The original checkpoint is never overwritten.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch


def prepare_checkpoint(source: dict, learning_rate: float = 1e-5) -> dict:
    result = deepcopy(source)
    # Critic prepends three privileged linear velocities before the actor's
    # proprioception. Its additional foot observations follow the twist slot.
    for key, width, yaw in (("actor_state_dict", 61, 50), ("critic_state_dict", 76, 53)):
        state = result[key]
        if state["mlp.0.weight"].shape[1] != width:
            raise ValueError(f"Unexpected {key} observation layout")
        for suffix in ("_mean", "_var", "_std"):
            if state[f"obs_normalizer.{suffix}"][0, yaw].item() != 0:
                raise ValueError(f"{key} is not a never-used zero-yaw checkpoint")
        # Old contribution was always exactly zero. New yaw values initially
        # have exactly zero contribution too, but this column can now learn.
        state["mlp.0.weight"][:, yaw] = 0
        state["obs_normalizer._std"][0, yaw] = 0.3
        state["obs_normalizer._var"][0, yaw] = 0.3 ** 2
    optimizer = result["optimizer_state_dict"]
    optimizer["state"] = {}
    for group in optimizer["param_groups"]:
        group["lr"] = learning_rate
    result["iter"] = 0
    result.setdefault("infos", {})
    if result["infos"] is None:
        result["infos"] = {}
    result["infos"]["env_state"] = {"common_step_counter": 0}
    result["infos"]["cliff_steering_preparation"] = {
        "actor_yaw_index": 50, "critic_yaw_index": 53,
        "yaw_normalizer_std": 0.3, "optimizer_reset": True,
        "learning_rate": learning_rate,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    prepared = prepare_checkpoint(source)
    prepared["infos"]["cliff_steering_preparation"].update({
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prepared, args.output)
    report = prepared["infos"]["cliff_steering_preparation"]
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
