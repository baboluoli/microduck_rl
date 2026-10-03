"""Approved Runpod experiment: preserve gait while learning heading correction.

200 updates maximum, 25-update gates, two-hour process deadline. Run this only
after approval; provisioning and stopping the pod are the operator's job.
"""

import json
import os
from pathlib import Path
import re
import time

import run_cliff_checkpoint_pilot as pilot


SOURCE = pilot.ROOT / "cliff-steering-seed/model_0.pt"
AGENT_ARGS = (
    "--agent.algorithm.learning-rate", "0.00001",
    "--agent.algorithm.schedule", "fixed",
    "--agent.algorithm.entropy-coef", "0.0",
    "--agent.algorithm.clip-param", "0.1",
)


def flat(checkpoints, name):
    directory = pilot.OUT / name
    pilot.run(["python", "scripts/diagnose_cliff_transfer.py", *map(str, checkpoints),
               "--terrain", "plane", "--device", "cuda:0", "--num-envs", "8",
               "--output", str(directory)], name)
    summaries = json.loads((directory / "summary.json").read_text())
    return {s["checkpoint"]: s["flat_upright"] / s["episodes"] for s in summaries}


def checked_train(source, count, name, num_envs):
    result = pilot.train(source, count, name, warm=False, num_envs=num_envs, agent_args=AGENT_ARGS)
    log = (pilot.OUT / f"{name}.log").read_text()
    nan_rates = re.findall(r"Episode_Termination/nan_state:\s*([\d.eE+-]+)", log)
    if not nan_rates or any(float(value) != 0 for value in nan_rates):
        raise RuntimeError(f"{name}: missing or nonzero NaN termination evidence")
    if re.search(r"(?i)(?:loss|reward|noise std)[^\n:]*:\s*(?:nan|inf)\b", log):
        raise RuntimeError(f"{name}: nonfinite training metric")
    return result


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError("First run scripts/prepare_cliff_steering_checkpoint.py")
    pilot.OUT = Path("logs/cliff-steering-pilot")
    pilot.PROFILE = "steering"
    pilot.DEADLINE = time.monotonic() + 2 * 60 * 60
    # An operator may reserve time for downloading and shutdown inside the
    # approved pod budget. This can shorten, never extend, the two-hour cap.
    if "MICRODUCK_CLIFF_POD_DEADLINE" in os.environ:
        remaining = float(os.environ["MICRODUCK_CLIFF_POD_DEADLINE"]) - time.time()
        if remaining <= 0:
            raise RuntimeError("The approved pod deadline has already elapsed")
        pilot.DEADLINE = min(pilot.DEADLINE, time.monotonic() + remaining)
    pilot.OUT.mkdir(parents=True, exist_ok=False)
    candidates = []
    reason = "Experiment interrupted; consult process log."
    try:
        baseline = pilot.evaluate([SOURCE], "baseline", no_videos=True)[str(SOURCE)]
        flat_baseline = flat([SOURCE], "flat-baseline")[str(SOURCE)]
        candidates.append({"updates": 0, "checkpoint": str(SOURCE), "rates": baseline,
                           "flat_upright": flat_baseline, "eligible": True})
        smoke = checked_train(SOURCE, 5, "cliff-steering-smoke", 64)
        pilot.run(["python", "scripts/export.py", pilot.TASK, "--checkpoint-file", str(smoke),
                   "--num-envs", "1", "--onnx-file", str(pilot.OUT / "smoke.onnx")], "smoke-export")
        previous = SOURCE
        reason = "Reached the 200-update cap; height remains 2 cm."
        for updates in range(25, 201, 25):
            checkpoint = checked_train(previous, 25, f"cliff-steering-update{updates}", 2048)
            rates = pilot.evaluate([previous, checkpoint], f"eval-update{updates}", no_videos=True)
            flat_rates = flat([previous, checkpoint], f"flat-update{updates}")
            measured = rates[str(checkpoint)]
            flat_rate = flat_rates[str(checkpoint)]
            regressed = (
                flat_rate < max(flat_baseline, flat_rates[str(previous)]) - 0.15
                or any(measured[seed][metric] < max(baseline[seed][metric], rates[str(previous)][seed][metric]) - 0.15
                       for seed in baseline for metric in ("complete", "edge", "reach"))
            )
            candidates.append({"updates": updates, "checkpoint": str(checkpoint), "rates": measured,
                               "flat_upright": flat_rate, "eligible": not regressed})
            if regressed:
                reason = f"Stopped at {updates}: >15 percentage-point regression in walking or cliff evaluation."
                break
            if updates >= 100 and max(c["rates"]["all"]["complete"] for c in candidates[1:]) <= baseline["all"]["complete"]:
                reason = f"Stopped at {updates}: no measured completion gain over the source."
                break
            previous = checkpoint
        best = max((c for c in candidates if c["eligible"]),
                   key=lambda c: (c["rates"]["all"]["complete"], c["rates"]["all"]["edge"], c["flat_upright"]))
        selected = Path(best["checkpoint"])
        # Full 144-episode approach battery and actual-outcome-labelled videos.
        pilot.evaluate([SOURCE] if selected == SOURCE else [SOURCE, selected], "final")
        pilot.run(["python", "scripts/export.py", pilot.TASK, "--checkpoint-file", str(selected),
                   "--num-envs", "1", "--onnx-file", str(pilot.OUT / "selected.onnx")], "selected-export")
    except Exception as exc:
        reason = f"Stopped: {type(exc).__name__}: {exc}"
        raise
    finally:
        eligible = [c for c in candidates if c["eligible"]]
        best = max(eligible, key=lambda c: (c["rates"]["all"]["complete"], c["rates"]["all"]["edge"], c["flat_upright"])) if eligible else None
        result = {"reason": reason, "profile": "steering", "height_cm": 2,
                  "selected": best, "candidates": candidates}
        (pilot.OUT / "pilot-result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
