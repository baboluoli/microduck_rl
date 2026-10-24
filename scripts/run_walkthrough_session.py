"""One approved pilot on the pod; no provisioning or account credentials.

Exit 20 means the frozen reference has no reachable partial-credit signal.
The caller must retrieve evidence and stop the pod in that case.
"""
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import signal
import sys
import subprocess
import time

TASK = 'Mjlab-CliffOracleWalkThrough-Fixed-MicroDuck'
SHA = '4b1eacacdab3773c8000a47416a30ed693d8d141915c258e2b0948b59702c651'
OUT = Path('logs/walkthrough-session')
ROOT = Path('logs/rsl_rl/oracle_walkthrough_fixed')
SOURCE = ROOT/'source-oracle500/model_500.pt'
DEADLINE = time.monotonic() + min(100*60, float(os.environ.get('MICRODUCK_WORK_DEADLINE', time.time()+100*60))-time.time())  # leave transfer/shutdown time inside 2h


def command(args, label, warm=False):
    env = dict(os.environ, WANDB_MODE='offline', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
               MICRODUCK_WARM_START='1' if warm else '0', PYTHONUNBUFFERED='1')
    with (OUT/f'{label}.log').open('w') as log:
        p = subprocess.Popen(['uv', 'run', *args], env=env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = p.wait(timeout=max(1, DEADLINE-time.monotonic()))
            if code: raise RuntimeError(f'{label} exit {code}')
        except BaseException:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            raise
    return (OUT/f'{label}.log').read_text()


def train(count, n, name):
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    text = command(['train', TASK, '--env.scene.num-envs', str(n),
        '--agent.max-iterations', str(count), '--agent.save-interval', '50', '--agent.seed', '42',
        '--agent.run-name', name, '--agent.resume', 'True', '--agent.load-run', 'source-oracle500',
        '--agent.load-checkpoint', 'model_500.pt', '--enable-nan-guard', 'True'], name, warm=True)
    assert 'Loading model checkpoint from:' in text and str(SOURCE) in text
    assert 'Patch 5: WARM START from' in text and 'weights/normalizer/optimizer kept' in text
    assert not re.search(r'(?i)(loss|reward)[^\n]{0,80}\b(?:nan|inf)\b', text)
    assert '[NanGuard] NaN/Inf found' not in text
    directory = sorted(ROOT.glob(f'*_{name}'))[-1]
    return sorted(directory.glob('model_*.pt'), key=lambda p:int(p.stem.split('_')[-1]))


def evaluate(checkpoint, name, seeds=(20,21,22)):
    command(['python', 'scripts/eval_cliff_walkthrough.py', str(checkpoint),
             '--output', str(OUT/name), '--seeds', *map(str,seeds)], name)
    rows = json.loads((OUT/name/'episodes.json').read_text())
    return score_rows(rows, checkpoint)


def score_rows(rows, checkpoint):
    assert len(rows) == 72
    def metrics(group):
        assert len(group) in (24, 72)
        n = len(group)
        prefixes = [r['prefix']['clean_prefix_blocks'] for r in group]
        return dict(success=sum(r['sustained']['success'] for r in group)/n,
                    capped_mean=sum(min(x,5) for x in prefixes)/n,
                    impact=sum('cliff_failed_landing' in r['termination'] for r in group)/n,
                    landing=sum(r.get('diagnostic', {}).get('original_goal_time_s',
                        r['sustained']['milestone_time_s']) is not None for r in group)/n,
                    milestone=sum(r['sustained']['milestone_time_s'] is not None for r in group)/n,
                    boundary=sum('boundary' in r['termination'] for r in group)/n,
                    nonfinite=sum('nan' in r['termination'] or 'nonfinite' in r['sustained']['reasons'] for r in group),
                    histogram=[sum(min(x,5)==b for x in prefixes) for b in range(6)],
                    cumulative=[sum(x>=b for x in prefixes)/n for b in range(1,6)])
    return dict(checkpoint=str(checkpoint), **metrics(rows),
                by_seed={str(seed):metrics([r for r in rows if r['seed']==seed])
                         for seed in sorted({r['seed'] for r in rows})})


def rank(candidate):
    # Strict > comparison preserves the earlier checkpoint on exact ties.
    return (candidate['success'], candidate['capped_mean'], -candidate['impact'], candidate['landing'])


def breaches(candidate, best, source):
    def lower(metric, pooled, per_seed, reference):
        return (reference[metric]-candidate[metric] >= pooled-1e-12
                and sum(reference['by_seed'][s][metric]-candidate['by_seed'][s][metric]
                        >= per_seed-1e-12 for s in reference['by_seed']) >= 2)
    def higher(metric, pooled, per_seed):
        return (candidate[metric]-source[metric] >= pooled-1e-12
                and sum(candidate['by_seed'][s][metric]-source['by_seed'][s][metric]
                        >= per_seed-1e-12 for s in source['by_seed']) >= 2)
    return dict(strict=lower('success',.15,.10,best),
                walking=lower('capped_mean',.50,.25,source),
                impact=higher('impact',.15,.10))


def update_streaks(streaks, flags):
    return {name:streaks.get(name,0)+1 if value else 0 for name,value in flags.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile('source-oracle500.pt', SOURCE)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    command(['--with', 'pytest', 'pytest', 'tests/test_cliff_walkthrough.py', 'tests/test_cliff_oracle.py',
             'tests/test_cliff_eval_videos.py', 'tests/test_cliff_oracle_diagnostic.py',
             'tests/test_walkthrough_session.py', 'tests/test_walkthrough_block_analysis.py', '-q'], 'tests')
    reference = evaluate(SOURCE, 'reference')
    (OUT/'reference-summary.json').write_text(json.dumps(reference,indent=2))
    command(['python', 'scripts/render_cliff_walkthrough.py', str(SOURCE),
             '--output', str(OUT/'reference-seed20-env0.mp4')], 'reference-video')
    if reference['capped_mean'] == 0:
        (OUT/'result.json').write_text(json.dumps({'stop':'zero clean walking blocks in frozen reference; pilot not launched', 'reference':reference},indent=2))
        return 20
    # The controller records a scale review before allowing any PPO updates.
    # This is an experiment audit gate, not a request for another run.
    while not (OUT/'reward-audit-reviewed.json').exists():
        if time.monotonic() >= DEADLINE:
            raise TimeoutError('Reward audit was not reviewed before work deadline')
        time.sleep(2)
    smoke = train(5, 64, 'walkthrough-fixed-smoke')[-1]
    command(['python', 'scripts/export.py', TASK, '--checkpoint-file', str(smoke),
             '--num-envs', '1', '--onnx-file', str(OUT/'smoke.onnx')], 'smoke-export')
    import onnxruntime as ort
    import numpy as np
    session = ort.InferenceSession(str(OUT/'smoke.onnx'), providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape[-1] == 63
    output = session.run(None, {session.get_inputs()[0].name: np.zeros((1,63), dtype=np.float32)})[0]
    assert output.shape == (1,14) and np.isfinite(output).all()
    # One uninterrupted pilot; evaluation checks are spaced at least seven
    # minutes apart. The final checkpoint is always evaluated on completion.
    env = dict(os.environ, WANDB_MODE='offline', MICRODUCK_WARM_START='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', PYTHONUNBUFFERED='1')
    args = ['uv','run','train',TASK,'--env.scene.num-envs','2048','--agent.max-iterations','250',
            '--agent.save-interval','50','--agent.seed','42','--agent.run-name','walkthrough-fixed-seed42',
            '--agent.resume','True','--agent.load-run','source-oracle500','--agent.load-checkpoint','model_500.pt',
            '--enable-nan-guard','True']
    candidates = [reference]
    best = reference
    streaks = dict(strict=0, walking=0, impact=0)
    seen = set()
    files = []
    final_checked = False
    next_check = time.monotonic()+420
    with (OUT/'pilot.log').open('w') as log:
        p = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        numeric_watch = subprocess.Popen([sys.executable, 'scripts/watch_walkthrough_numerics.py',
            str(p.pid), str(OUT/'pilot.log'), str(OUT/'numerical-stop.json')])
        try:
            while True:
                # Safety monitoring is separate from the seven-minute behavior
                # evaluation cadence: numerical failures stop immediately.
                log.flush()
                recent = (OUT/'pilot.log').read_text()
                if '[NanGuard] NaN/Inf found' in recent or re.search(r'(?i)(loss|reward)[^\n]{0,80}\b(?:nan|inf)\b', recent):
                    raise RuntimeError('Nonfinite training state/loss; stopping immediately')
                finished = p.poll() is not None
                if time.monotonic() >= DEADLINE: raise TimeoutError('Session work cap')
                if finished and p.returncode:
                    raise RuntimeError(f'Pilot exit {p.returncode}')
                if time.monotonic() >= next_check or (finished and not seen):
                    dirs = sorted(ROOT.glob('*_walkthrough-fixed-seed42'))
                    files = [] if not dirs else sorted(dirs[-1].glob('model_*.pt'), key=lambda x:int(x.stem.split('_')[-1]))
                    if files:
                        checkpoint = files[-1]
                        # Saving can overlap a periodic check. Prove the file
                        # is loadable before evaluating; completion has closed it.
                        import torch
                        try: torch.load(checkpoint, map_location='cpu', weights_only=False)
                        except (EOFError, RuntimeError): checkpoint = None
                        if checkpoint is not None and str(checkpoint) not in seen:
                            seen.add(str(checkpoint))
                            measured = evaluate(checkpoint, f'eval-{checkpoint.stem}')
                            candidates.append(measured)
                            if measured['nonfinite']:
                                raise RuntimeError('Nonfinite evaluation evidence')
                            flags = breaches(measured, best, reference)
                            streaks = update_streaks(streaks, flags)
                            if rank(measured) > rank(best): best = measured
                            (OUT/'progress.json').write_text(json.dumps({'best':best,'candidates':candidates,
                                'consecutive_regressions':streaks, 'breaches':flags},indent=2))
                            if max(streaks.values()) >= 2: break
                        if finished and checkpoint is not None and str(checkpoint) in seen:
                            final_checked = True
                    next_check = time.monotonic()+420
                if final_checked:
                    if p.returncode: raise RuntimeError(f'Pilot exit {p.returncode}')
                    break
                time.sleep(1)
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=30)
            if numeric_watch.poll() is None:
                numeric_watch.terminate()
            numeric_watch.wait(timeout=10)
    text = (OUT/'pilot.log').read_text()
    assert 'Loading model checkpoint from:' in text and str(SOURCE) in text
    assert 'Patch 5: WARM START from' in text and 'weights/normalizer/optimizer kept' in text
    command(['python', 'scripts/export.py', TASK, '--checkpoint-file', best['checkpoint'],
             '--num-envs', '1', '--onnx-file', str(OUT/'best.onnx')], 'best-export')
    # Held-out seeds 30–32 remain reserved for a separately reviewed finalist.
    if best is not reference:
        command(['python', 'scripts/render_cliff_walkthrough.py', best['checkpoint'],
                 '--output', str(OUT/'best-seed20-env0.mp4')], 'best-video')
    (OUT/'result.json').write_text(json.dumps({'best':best,'candidates':candidates,'early_stop':max(streaks.values())>=2,'streaks':streaks},indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
