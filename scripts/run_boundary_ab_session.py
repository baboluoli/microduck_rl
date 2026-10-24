"""Boundary-semantics A/B on an approved, preflighted pod (does not provision one).

One session does two things:

1. Telemetry evaluations (heading, command, body/world velocities) of the
   unchanged source 249 and of the old drifting ``model_99`` on exactly the
   ``eval-005``/``eval-006`` battery: heights 2/2.25 cm x starts 45/55/65 cm x
   seeds 20/21/22 x 24 envs. These answer turning versus sideways walking.
2. A 100-update continuation from the same source, with the same staged
   curriculum state (upper 22.5 mm), seed, env count and warm start as the run
   that produced ``model_99``. The only intended behavioral difference is that
   course-boundary exits are now terminal failures rather than timeouts. The
   result is then evaluated on the same battery.

Training runs with the NaN guard and ``nan_forensics``; any NaN stops the
session immediately and the dumps are copied into the output. An external
watchdog must still stop the pod if this process or SSH dies.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import torch

from cliff_course_process import wait_checked
from mjlab_microduck.tasks.cliff.course_curriculum import Curriculum

TASK = 'Mjlab-CliffOracleCourse-MicroDuck'
ROOT = Path('logs/rsl_rl/oracle_course')
HEIGHTS = (.02, .0225)
DISTANCES = (.45, .55, .65)
SEEDS = (20, 21, 22)
SOURCE_SHA = 'ef70ae031f61b64686aef2d428338de5d2562adf0d6847b3d133092770223a06'
OLD_SHA = 'ce9f08bbb12a102d5ba08a2d5f14fd5d90a32e38c5800f037d0e0351349d557a'
NAN_DUMPS = Path('/tmp/mjlab/nan_dumps')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(directory):
    """Successes, boundary exits and mean final Y per policy battery."""
    rows = [r for p in sorted(directory.glob('h*-d*-s*.json')) for r in json.loads(p.read_text())]
    out = dict(episodes=len(rows), successes=sum(bool(r['success']) for r in rows))
    outcomes = {}
    for r in rows:
        outcomes[str(r.get('outcome'))] = outcomes.get(str(r.get('outcome')), 0) + 1
    out['outcomes'] = outcomes
    edge_y = []
    for r in rows:
        trace = r.get('trace') or []
        crossing = next((s for s in trace if s['x'] >= .55), None)
        if crossing is not None:
            edge_y.append(crossing['y'])
    out['edge_crossings'] = len(edge_y)
    out['mean_edge_y_m'] = sum(edge_y) / len(edge_y) if edge_y else None
    out['negative_y_at_edge'] = sum(y < 0 for y in edge_y)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Unchanged continuation 249 (model_source.pt)')
    p.add_argument('--old-trained', type=Path, required=True, help='Old model_99 (boundary as timeout)')
    p.add_argument('--curriculum', type=Path, required=True, help='curriculum.json of the 2026-10-24 session')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline-epoch', type=float, required=True, help='Absolute approved deadline, including retrieval')
    p.add_argument('--updates', type=int, default=100)
    args = p.parse_args()
    assert sha(args.source) == SOURCE_SHA, 'Source checkpoint is not continuation 249'
    assert sha(args.old_trained) == OLD_SHA, 'Old trained checkpoint is not the audited model_99'
    # Reserve 15 minutes for retrieval/shutdown; never exceed two hours.
    deadline = min(args.deadline_epoch, time.time() + 7200) - 900
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'session.json').write_text(json.dumps(dict(
        source=str(args.source), source_sha256=SOURCE_SHA, old_trained=str(args.old_trained),
        old_trained_sha256=OLD_SHA, deadline_epoch=args.deadline_epoch, training_seed=42,
        updates=args.updates, training_envs=2048, heights=HEIGHTS, distances=DISTANCES, seeds=SEEDS,
        retrieval_reserve_s=900), indent=2))
    (args.output / 'source.diff').write_bytes(subprocess.run(
        ['git', 'diff', 'HEAD'], capture_output=True).stdout)
    environment = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', WANDB_MODE='offline', PYTHONUNBUFFERED='1')

    def command(cmd, label, warm=False):
        if time.time() >= deadline:
            raise TimeoutError('Retrieval reserve reached')
        log_path = args.output / f'{label}.log'
        with log_path.open('w') as log:
            process = subprocess.Popen(['uv', 'run', *cmd], stdout=log, stderr=subprocess.STDOUT,
                env=dict(environment, MICRODUCK_WARM_START='1' if warm else '0'), start_new_session=True)
            try:
                code = wait_checked(process, log_path, deadline)
                if code:
                    raise RuntimeError(f'{label} exit {code}')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)  # Let forensics finish writing.
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        text = log_path.read_text()
        if re.search(r'(?i)(loss|reward)[^\n]{0,80}\b(?:nan|inf)\b', text):
            raise RuntimeError('Nonfinite losses or rewards')
        if '[NanGuard] NaN/Inf found' in text:
            raise RuntimeError('Nonfinite training state')
        return text

    summaries = {}

    def evaluate(checkpoint, label):
        directory = args.output / label
        command(['python', 'scripts/eval_cliff_course.py', str(checkpoint), '--output', str(directory),
                 '--heights', *map(str, HEIGHTS), '--distances', *map(str, DISTANCES),
                 '--seeds', *map(str, SEEDS)], label)
        for path in directory.glob('h*-d*-s*.json'):
            rows = json.loads(path.read_text())
            assert len(rows) == 24, path
            if any('nonfinite' in r['events'] for r in rows):
                raise RuntimeError(f'Nonfinite evaluation state in {path}')
        summaries[label] = summarize(directory)
        (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2))

    try:
        command(['--with', 'pytest', 'pytest', 'tests/test_cliff_course.py', 'tests/test_nan_forensics.py', '-q'], 'runtime-tests')
        # Telemetry first: cheap, and useful even if training later fails.
        evaluate(args.source, 'eval-source')
        evaluate(args.old_trained, 'eval-old-model99')

        state = Curriculum.load(args.curriculum)
        assert state.upper_mm == 22.5, state.upper_mm
        ROOT.mkdir(parents=True, exist_ok=True)
        source_dir = ROOT / 'boundary-ab-source'
        source_dir.mkdir(exist_ok=False)
        copied = source_dir / 'model_source.pt'
        shutil.copy2(args.source, copied)

        def stage(checkpoint, iteration):
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            payload['iter'] = iteration
            payload['infos'] = dict(payload.get('infos') or {})
            payload['infos']['course_curriculum'] = asdict(state)
            payload['infos']['env_state'] = {'common_step_counter': iteration * 24}
            torch.save(payload, source_dir / 'model_staged.pt')

        def train(limit, envs, label):
            stage(copied, 0)
            text = command(['train', TASK, '--env.scene.num-envs', str(envs), '--agent.max-iterations', str(limit),
                '--agent.save-interval', '100', '--agent.seed', '42', '--agent.run-name', label,
                '--agent.resume', 'True', '--agent.load-run', 'boundary-ab-source',
                '--agent.load-checkpoint', 'model_staged.pt', '--enable-nan-guard', 'True'], label, warm=True)
            assert 'Loading model checkpoint from:' in text
            assert 'weights/normalizer/optimizer kept' in text
            directory = sorted(ROOT.glob(f'*_{label}'))[-1]
            return max(directory.glob('model_*.pt'), key=lambda q: int(q.stem.split('_')[-1]))

        smoke = train(5, 64, 'boundary-ab-smoke')
        command(['python', 'scripts/export.py', TASK, '--checkpoint-file', str(smoke),
                 '--num-envs', '1', '--onnx-file', str(args.output / 'smoke.onnx')], 'smoke-export')
        trained = train(args.updates, 2048, f'boundary-ab-{args.updates}')
        (args.output / 'trained.json').write_text(json.dumps(dict(checkpoint=str(trained), sha256=sha(trained))))
        evaluate(trained, f'eval-new-model{args.updates - 1}')
        (args.output / 'complete.json').write_text(json.dumps(dict(finished_unix=time.time())))
    except Exception as error:
        (args.output / 'stopped.json').write_text(json.dumps(dict(reason=repr(error), at_unix=time.time())))
        raise
    finally:
        if NAN_DUMPS.exists():
            shutil.copytree(NAN_DUMPS, args.output / 'nan-dumps', symlinks=True, dirs_exist_ok=True)
        files = [q for q in args.output.rglob('*') if q.is_file() and not q.is_symlink()]
        files += [q for d in ROOT.glob('*boundary-ab*') for q in d.rglob('*') if q.is_file()]
        (args.output / 'SHA256SUMS').write_text(''.join(
            f'{sha(q)}  {q}\n' for q in files if q.name != 'SHA256SUMS'))


if __name__ == '__main__':
    main()
