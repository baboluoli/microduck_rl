"""Run an approved, offline 2 cm pilot from the local model_999 checkpoint.

Invoke on a Linux GPU pod, from the repository root, after uv sync. This script
does not provision resources. Runpod launches still require user approval.
"""

import csv
import json
import os
from pathlib import Path
import subprocess
import time
import signal


TASK = "Mjlab-CliffDrop-Flat-MicroDuck"
ROOT = Path("logs/rsl_rl/cliff_drop")
SOURCE = ROOT / "2026-09-26_20-12-42_cliff-drop-first-1000/model_999.pt"
OUT = Path("logs/cliff-model999-pilot")
PROFILE = "model999"
DEADLINE = None


def run(args, name, *, warm=False):
    env = dict(os.environ, WANDB_MODE="offline", MICRODUCK_CLIFF_PROFILE=PROFILE,
               MICRODUCK_CLIFF_MAX_HEIGHT_CM="2", MICRODUCK_WARM_START="1" if warm else "0",
               MUJOCO_GL="egl", PYTHONUNBUFFERED="1")
    print(f"Starting {name}", flush=True)
    with (OUT / f"{name}.log").open("w") as log:
        # Kill the entire uv/python process group on timeout or interruption;
        # killing just uv can otherwise leave GPU training running orphaned.
        with subprocess.Popen(["uv", "run", *args], env=env, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                code = process.wait(timeout=None if DEADLINE is None else max(1, DEADLINE - time.monotonic()))
                if code:
                    raise subprocess.CalledProcessError(code, process.args)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
    print(f"Finished {name}", flush=True)


def train(source, count, name, *, warm, num_envs, agent_args=()):
    run(["train", TASK, "--env.scene.num-envs", str(num_envs),
         "--agent.max-iterations", str(count), "--agent.save-interval", "100",
         "--agent.run-name", name, "--agent.resume", "True",
         "--agent.load-run", source.parent.name,
         "--agent.load-checkpoint", source.name, *agent_args], name, warm=warm)
    log = (OUT / f"{name}.log").read_text()
    if f"[INFO]: Loading model checkpoint from: {source}" not in log:
        raise RuntimeError(f"{name} did not confirm checkpoint loading")
    if warm and ("Patch 5: WARM START from" not in log or
                 "weights/normalizer/optimizer kept" not in log):
        raise RuntimeError(f"{name} did not confirm warm-start counter reset")
    directory = sorted(ROOT.glob(f"*_{name}"))[-1]
    return max(directory.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))


def evaluate(checkpoints, name, *, no_videos=False):
    run(["python", "scripts/eval_cliff_drop.py", *map(str, checkpoints),
         "--heights-cm", "2", "--num-envs", "16", "--include-midair",
         "--output", str(OUT / name), *(["--no-videos"] if no_videos else [])], name)
    report = json.loads((OUT / f"{name}.json").read_text())
    if report.get("video_errors"):
        raise RuntimeError(f"{name} video rendering failed: {report['video_errors']}")
    with (OUT / f"{name}.csv").open() as file:
        rows = list(csv.DictReader(file))
    rates = {}
    for checkpoint in checkpoints:
        sample = [r for r in rows if r["checkpoint"] == str(checkpoint) and r["start"] == "upper"]
        rates[str(checkpoint)] = {}
        for seed in ("0", "1", "2", "all"):
            group = [r for r in sample if seed == "all" or r["seed"] == seed]
            if not group:
                raise RuntimeError(f"Missing evaluation episodes for seed {seed}")
            rates[str(checkpoint)][seed] = {
                "complete": sum(r["outcome"] == "complete" for r in group) / len(group),
                "edge": sum(r["edge_in_corridor"] == "True" for r in group) / len(group),
                "reach": sum(r["reached_edge"] == "True" for r in group) / len(group),
            }
    print(json.dumps(rates), flush=True)
    return rates


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=False)
    baseline = evaluate([SOURCE], "baseline")[str(SOURCE)]
    smoke = train(SOURCE, 5, "cliff-model999-smoke", warm=True, num_envs=64)
    run(["python", "scripts/export.py", TASK, "--checkpoint-file", str(smoke),
         "--num-envs", "1", "--onnx-file", str(OUT / "smoke.onnx")], "smoke-export")
    candidates = [{"updates": 0, "checkpoint": str(SOURCE), "rates": baseline}]
    previous = SOURCE
    updates = 0
    reason = "Completed the 250-update pilot; height remains 2 cm."
    for count in (100, 100, 50):
        checkpoint = train(previous, count, f"cliff-model999-update{updates + count}",
                           warm=updates == 0, num_envs=2048)
        updates += count
        rates = evaluate([previous, checkpoint], f"eval-update{updates}")
        measured = rates[str(checkpoint)]
        candidates.append({"updates": updates, "checkpoint": str(checkpoint), "rates": measured})
        # A small battery has sampling noise. Stop for a >15 percentage-point
        # drop against the source or previous block, in any seed or pooled rate.
        reference = rates[str(previous)]
        if any(measured[seed][metric] < max(baseline[seed][metric], reference[seed][metric]) - 0.15
               for seed in baseline for metric in ("complete", "edge", "reach")):
            reason = f"Stopped at {updates} updates: evaluation regressed by >15 percentage points."
            break
        previous = checkpoint
    # Prefer the baseline on a tie; midair starts never determine selection.
    best = max(candidates, key=lambda c: (c["rates"]["all"]["complete"],
                                         c["rates"]["all"]["edge"],
                                         c["rates"]["all"]["reach"]))
    result = {"reason": reason, "selected": best, "candidates": candidates,
              "profile": "model999", "height_cm": 2,
              "regression_stop_threshold_percentage_points": 15}
    (OUT / "pilot-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
