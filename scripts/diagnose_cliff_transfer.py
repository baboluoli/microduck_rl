"""Compare fixed checkpoints on a plane and a 2 cm ledge, without learning."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import torch

from eval_cliff_drop import evaluate
from mjlab.utils.torch import configure_torch_backends


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--terrain", nargs="+", choices=("plane", "cliff"), default=["plane", "cliff"])
    parser.add_argument("--output", type=Path, default=Path("logs/cliff-transfer-diagnostic"))
    args = parser.parse_args()
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.set_num_threads(1)
    import mjlab.tasks  # noqa: F401

    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    all_rows = []
    for checkpoint in args.checkpoints:
        for terrain in args.terrain:
            flat = terrain == "plane"
            rows = []
            for seed in args.seeds:
                rows.extend(evaluate(checkpoint, num_envs=args.num_envs, seed=seed,
                                     device=args.device, drop_m=0.02, approach_m=0.55,
                                     flat_ground=flat))
            all_rows.extend(rows)
            summary = {"checkpoint": str(checkpoint), "terrain": "plane" if flat else "cliff",
                       "episodes": len(rows), "flat_upright": sum(r["outcome"] == "flat_upright" for r in rows),
                       "complete": sum(r["outcome"] == "complete" for r in rows),
                       "reached_edge": sum(r["reached_edge"] for r in rows),
                       "head_impacts": sum(r["max_head_force_n"] > 15 for r in rows),
                       "median_forward_m": statistics.median(r["forward_progress_m"] for r in rows),
                       "median_path_m": statistics.median(r["distance_traveled_m"] for r in rows),
                       "median_heading_error_deg": statistics.median(r["mean_heading_error_deg"] for r in rows),
                       "median_peak_torque_nm": statistics.median(r["peak_torque_nm"] for r in rows),
                       "outcomes": {k: sum(r["outcome"] == k for r in rows)
                                    for k in sorted({r["outcome"] for r in rows})}}
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
            # Persist after each condition so an interrupted diagnostic remains useful.
            (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            with (args.output / "episodes.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=all_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_rows)


if __name__ == "__main__":
    main()
