"""A success selected for replay must not mislabel a failed recording."""

import json
import importlib.util
from pathlib import Path
import sys

_spec = importlib.util.spec_from_file_location(
    "eval_cliff_drop", Path(__file__).resolve().parents[1] / "scripts/eval_cliff_drop.py"
)
eval_cliff_drop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_cliff_drop)


def test_video_reports_actual_replay_outcome(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model_999.pt"
    output = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["eval_cliff_drop.py", str(checkpoint),
                                     "--heights-cm", "2", "--seeds", "0",
                                     "--approach-distances", "0.55",
                                     "--output", str(output)])
    monkeypatch.setattr(eval_cliff_drop, "configure_torch_backends", lambda: None)

    def evaluate(path, **kwargs):
        video = kwargs.get("video_path")
        if video is not None:
            video.write_bytes(b"recorded-video-placeholder")
        return [{"env": 0, "checkpoint": str(path), "start": "upper",
                 "seed": 0, "height_cm": 2, "approach_m": 0.55,
                 "outcome": "before_edge" if video else "complete",
                 "edge_in_corridor": video is None, "lower_contact": video is None}]

    monkeypatch.setattr(eval_cliff_drop, "evaluate", evaluate)
    eval_cliff_drop.main()
    result = json.loads(output.with_suffix(".json").read_text())
    assert result["summary"][str(checkpoint)]["upper"]["2"]["complete"] == 1
    recorded = result["video_results"][0]
    assert recorded["selected_from"] == "success"
    assert recorded["episode"]["outcome"] == "before_edge"
    assert "-before_edge-" in recorded["path"]
    assert Path(recorded["path"]).exists()
    assert not list(tmp_path.glob("*-success-*.mp4"))
