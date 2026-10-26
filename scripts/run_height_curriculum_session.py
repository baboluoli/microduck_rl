"""Gated height curriculum toward 14 cm, on an approved, preflighted pod.

Does not provision a pod. An external watchdog must still stop the pod if this
process or SSH dies.

What one session does:

1. GPU runtime tests, then a 5-update smoke run at 64 envs plus ONNX export,
   with the curriculum staged at its current top height. Then a baseline: the
   source checkpoint gets the same development check (``eval-000-*``, recorded
   as ``block: 0`` in ``stages.jsonl``), so later blocks have a reference.
2. A loop of 100-update training blocks (2048 envs, seed 42). After every
   block a fixed-cost check runs on development seeds 20–22:
   - frontier: the current top height at starts 45/55/65 cm;
   - regression: 2 cm at 55 cm, plus the previous stage at 55 cm.
   If the whole frontier passes, a confirmation repeats it on seeds 40–42.
   Passing both raises the height one stage (``course.STAGES``) and records
   the checkpoint as ``best`` for the stage it just mastered.
3. Stops at the last stage, when a regression condition fails two checks in
   a row, when a stage is still not passed after ``--max-blocks-per-stage``
   blocks (a stall), or when there is not enough time left for another block
   plus its checks (``provisional.json``).

``--resume-source`` treats the source as the SAME task (a resume, not a warm
start): block 1 and the smoke run keep the source's iteration and step
counter, so step-based schedules (the course-walk smoothing and DR ramps)
stay where the source finished them. Each training log must then show the
counter that was staged, at or past every schedule's final stage.

``--hold-height`` keeps the height fixed for ``--blocks`` blocks: the checks
run as usual and record ``would_advance``, but the stage is never raised and
there is no stall stop. The regression stop still applies.

Every ``stages.jsonl`` entry carries ``quality``: action std plus the speed,
wobble and time-to-goal numbers from ``summarize_height_curriculum.walking_quality``.

After every check it saves ``curriculum.json`` and ``selection.json`` and
appends one line to ``stages.jsonl``. The next session continues from those.
Training time alone never raises the height.
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

from mjlab_microduck.tasks.cliff import course
from mjlab_microduck.tasks.cliff.course_curriculum import (
    DISTANCES, REGRESSION_DISTANCE, Curriculum, Result, key)

# Training task -> its rsl_rl log directory (the RL cfg's experiment_name).
# Evaluation always uses eval_cliff_course.py's fixed 12 s course rule.
TASKS = {
    'course': ('Mjlab-CliffOracleCourse-MicroDuck', Path('logs/rsl_rl/oracle_course')),
    # No finish, no completion credit, velocity tracking, 8 s episodes.
    'walk': ('Mjlab-CliffOracleCourseWalk-MicroDuck', Path('logs/rsl_rl/oracle_course_walk')),
}
TASK, ROOT = TASKS['course']
DEVELOPMENT_SEEDS = (20, 21, 22)
CONFIRMATION_SEEDS = (40, 41, 42)
NAN_DUMPS = Path('/tmp/mjlab/nan_dumps')
STEPS_PER_UPDATE = 24  # NUM_STEPS_PER_ENV: env steps per PPO update.
# Measured on an A40 (2026-10-24): 100 updates ~4.5 min, development check
# (5 conditions x 3 seeds) ~8 min, confirmation (3 x 3) ~5 min. Do not start
# a block that cannot finish together with both checks.
BLOCK_BUDGET_S = 17 * 60
RETRIEVAL_RESERVE_S = 15 * 60
SESSION_CAP_S = 2 * 3600


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sampled_bands(counts, upper):
    """Share of reset samples in the frontier band, and samples above the stage.

    ``counts`` is the per-row histogram the runner prints at each save. Any
    sample above the current top height means the sampler or the restored
    curriculum state is wrong.
    """
    easier_end, frontier_end = course.bands(upper)
    total = sum(counts)
    frontier = sum(counts[easier_end:frontier_end])
    return dict(frontier_share=frontier / total if total else None,
                above_stage=sum(counts[frontier_end:]), total=total)


def start_point(payload, resume):
    """(iteration to stage the source at, warm-start the first training call?).

    Default: a warm start from another task, counters restart at 0. Resume:
    the source's own iteration, so its schedules keep their stage.
    """
    return (int(payload['iter']), False) if resume else (0, True)


def final_schedule_step(curriculum):
    """Env step at which the last staged schedule reaches its final stage.

    ``curriculum`` is an env cfg's curriculum dict. Any term param that is a
    list of ``{'step': ...}`` dicts counts as a schedule; 0 if there are none.
    """
    steps = [stages[-1]['step'] for term in curriculum.values() for stages in (term.params or {}).values()
             if isinstance(stages, list) and stages and isinstance(stages[0], dict) and 'step' in stages[0]]
    return max(steps, default=0)


def resumed_counter(text):
    """The step counter a plain resume kept, from mdp.py Patch 5's log line."""
    found = re.findall(r'\[mdp\] Patch 5: RESUME from .* common_step_counter (\d+) kept', text)
    return int(found[-1]) if found else None


def run_blocks(state, *, train_block, check, confirm, now, deadline, record,
               max_blocks_per_stage=5, block_budget_s=BLOCK_BUDGET_S, hold=False, max_blocks=None,
               warm_first=True):
    """The block loop: train, check, maybe confirm and advance, decide to stop.

    Pure decision logic. ``train_block(block, warm)`` returns a checkpoint,
    ``check(checkpoint)`` returns development results (frontier + regression),
    ``confirm(checkpoint)`` the frontier on confirmation seeds. ``record`` is
    called after every check with the entry to save. Returns the stop reason.

    ``hold=True`` keeps the height fixed: the same checks run, but the stage
    is never raised. ``would_advance`` records what the gate would have done.
    The stall stop is off; the regression stop and time budget still apply.
    ``max_blocks`` stops the loop after that many blocks. ``warm_first=False``
    (a resume) trains block 1 without the warm start too.
    """
    block = 0
    while True:
        if max_blocks is not None and block >= max_blocks:
            return 'block limit'
        if deadline - now() < block_budget_s:
            return 'time budget'
        block += 1
        checkpoint = train_block(block, warm=warm_first and block == 1)
        state.blocks_at_stage += 1
        upper_mm = state.upper_mm
        development = check(checkpoint)
        regressed = state.regression(development)
        frontier_passed = all(development[k].passes for k in state.frontier())
        confirmation, advanced, would_advance = None, False, False
        if frontier_passed and not regressed:
            confirmation = confirm(checkpoint)
            if hold:
                # Same test as Curriculum.advance, without changing the state.
                would_advance = all(confirmation[k].passes for k in state.frontier())
            else:
                advanced = state.advance(development, confirmation, str(checkpoint))
                would_advance = advanced or state.mastered
        mastered = state.mastered
        entry = dict(block=block, checkpoint=str(checkpoint), upper_mm=upper_mm,
                     blocks_at_stage=state.blocks_at_stage if not advanced else
                     state.history[-1]['blocks'],
                     development={k: asdict(v) for k, v in development.items()},
                     development_passes={k: v.passes for k, v in development.items()},
                     regression_streaks=dict(state.regression_streaks),
                     confirmation=None if confirmation is None else
                     {k: asdict(v) for k, v in confirmation.items()},
                     confirmation_passes=None if confirmation is None else
                     {k: v.passes for k, v in confirmation.items()},
                     advanced=advanced, mastered_stage=advanced or mastered,
                     hold=hold, would_advance=would_advance,
                     next_upper_mm=state.upper_mm, at_unix=now())
        # In hold mode a passed gate still marks the checkpoint as ``best``
        # for this height: the candidate to restart the ladder from.
        passed = would_advance if hold else advanced or mastered
        record(entry, checkpoint, passed_stage=upper_mm if passed else None)
        if regressed:
            return 'regression'
        if mastered:
            return 'last stage mastered'
        if not hold and not advanced and state.stalled(max_blocks_per_stage):
            return 'stall'


def results(directory, heights, distances, seeds):
    """Read eval_cliff_course.py output into one Result per height/distance."""
    batch = {}
    for h in heights:
        for d in distances:
            groups = [json.loads((directory / f'h{h:.4f}-d{d:.2f}-s{s}.json').read_text()) for s in seeds]
            assert all(len(g) == 24 for g in groups), directory
            if any('nonfinite' in r['events'] for g in groups for r in g):
                raise RuntimeError(f'Nonfinite evaluation state in {directory}')
            batch[key(round(h * 1000, 1), round(d * 100))] = Result(
                tuple(sum(r['success'] for r in g) for g in groups),
                sum('impact' in r['events'] for g in groups for r in g),
                sum(r['one_metre'] for g in groups for r in g))
    return batch


def main():
    import torch

    from cliff_course_process import wait_checked

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, required=True, help='Checkpoint to continue (e.g. drift-penalty model_99.pt)')
    p.add_argument('--source-sha', required=True, help='Expected sha256 of --source; asserted')
    p.add_argument('--curriculum', type=Path, required=True, help='curriculum.json saved by the previous session')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline-epoch', type=float, required=True, help='Absolute approved deadline, including retrieval')
    p.add_argument('--max-nan-resets', type=int, default=5,
                   help='Training NaN events tolerated per block (the guard resets the broken env)')
    p.add_argument('--label', default='height-curriculum', help='Training run label prefix')
    p.add_argument('--block-updates', type=int, default=100)
    p.add_argument('--max-blocks-per-stage', type=int, default=5)
    p.add_argument('--hold-height', action='store_true',
                   help='Keep the current height: never advance, no stall stop (records would_advance)')
    p.add_argument('--blocks', type=int, default=4, help='With --hold-height: number of blocks to train')
    p.add_argument('--task', choices=TASKS, default='course', help='Training task (evaluation is unchanged)')
    p.add_argument('--resume-source', action='store_true',
                   help="Source is the same task: keep its iteration and step counter (no warm start)")
    args = p.parse_args()
    global TASK, ROOT
    TASK, ROOT = TASKS[args.task]
    assert sha(args.source) == args.source_sha, 'Source checkpoint sha256 does not match --source-sha'
    # Reserve 15 minutes for retrieval/shutdown; never exceed two hours.
    deadline = min(args.deadline_epoch, time.time() + SESSION_CAP_S) - RETRIEVAL_RESERVE_S
    state = Curriculum.load(args.curriculum)
    source_iteration, warm_source = start_point(
        torch.load(args.source, map_location='cpu', weights_only=False), args.resume_source)
    final_step = 0
    if args.resume_source:
        from mjlab.tasks.registry import load_env_cfg
        import mjlab_microduck.tasks  # noqa: F401  Registers the tasks.
        final_step = final_schedule_step(load_env_cfg(TASK).curriculum)
        assert source_iteration * STEPS_PER_UPDATE >= final_step, (
            f'Source iteration {source_iteration} is before the final schedule stage (step {final_step})')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'session.json').write_text(json.dumps(dict(
        source=str(args.source), source_sha256=args.source_sha, curriculum=str(args.curriculum),
        starting_upper_mm=state.upper_mm, starting_blocks_at_stage=state.blocks_at_stage,
        stages=course.STAGES, max_nan_resets=args.max_nan_resets, label=args.label,
        deadline_epoch=args.deadline_epoch, training_seed=42, training_envs=2048,
        block_updates=args.block_updates, max_blocks_per_stage=args.max_blocks_per_stage,
        hold_height=args.hold_height, blocks=args.blocks if args.hold_height else None,
        task=TASK, distances=DISTANCES, regression_distance=REGRESSION_DISTANCE,
        development_seeds=DEVELOPMENT_SEEDS, confirmation_seeds=CONFIRMATION_SEEDS,
        block_budget_s=BLOCK_BUDGET_S, retrieval_reserve_s=RETRIEVAL_RESERVE_S,
        resume_source=args.resume_source, source_iteration=source_iteration, final_schedule_step=final_step,
        git_head=os.environ.get('MICRODUCK_GIT_HEAD')), indent=2))
    (args.output / 'source.diff').write_bytes(subprocess.run(
        ['git', 'diff', 'HEAD'], capture_output=True).stdout)
    environment = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', WANDB_MODE='offline', PYTHONUNBUFFERED='1')

    def command(cmd, label, warm=False, max_nan_resets=0):
        if time.time() >= deadline:
            raise TimeoutError('Retrieval reserve reached')
        log_path = args.output / f'{label}.log'
        with log_path.open('w') as log:
            process = subprocess.Popen(['uv', 'run', *cmd], stdout=log, stderr=subprocess.STDOUT,
                env=dict(environment, MICRODUCK_WARM_START='1' if warm else '0'), start_new_session=True)
            try:
                code = wait_checked(process, log_path, deadline, max_nan_resets)
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
        if not max_nan_resets and '[NanGuard] NaN/Inf found' in text:
            raise RuntimeError('Nonfinite training state')
        return text

    count = 0
    last_frontier = {}

    def evaluate(checkpoint, heights, distances, seeds, kind, number=None):
        # ``number`` overrides the running count (the baseline is eval-000).
        nonlocal count
        if number is None:
            count += 1
        directory = args.output / f'eval-{count if number is None else number:03}-{kind}'
        if kind == 'frontier':
            last_frontier['directory'] = directory
        command(['python', 'scripts/eval_cliff_course.py', str(checkpoint), '--output', str(directory),
                 '--heights', *map(str, heights), '--distances', *map(str, distances),
                 '--seeds', *map(str, seeds)], directory.name)
        return results(directory, heights, distances, seeds)

    def check(checkpoint, number=None):
        upper = state.upper_mm / 1000
        batch = evaluate(checkpoint, [upper], [d / 100 for d in DISTANCES], DEVELOPMENT_SEEDS, 'frontier', number)
        regression = [round(float(k.split('/')[0]) / 1000, 4) for k in state.regression_conditions()]
        batch.update(evaluate(checkpoint, regression, [REGRESSION_DISTANCE / 100], DEVELOPMENT_SEEDS,
                              'regression', number))
        return batch

    def confirm(checkpoint):
        return evaluate(checkpoint, [state.upper_mm / 1000], [d / 100 for d in DISTANCES],
                        CONFIRMATION_SEEDS, 'confirmation')

    ROOT.mkdir(parents=True, exist_ok=True)
    source_dir = ROOT / f'{args.label}-source'
    source_dir.mkdir(exist_ok=False)
    copied = source_dir / 'model_source.pt'
    shutil.copy2(args.source, copied)

    def stage(checkpoint, iteration):
        """Stage a checkpoint with the CURRENT curriculum state and step counter.

        The training process restores ``course_curriculum`` on load, which is
        how a raised height reaches the sampler. ``common_step_counter`` keeps
        step-based schedules consistent with the cumulative iteration.
        """
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        payload['iter'] = iteration
        payload['infos'] = dict(payload.get('infos') or {})
        payload['infos']['course_curriculum'] = asdict(state)
        payload['infos']['env_state'] = {'common_step_counter': iteration * STEPS_PER_UPDATE}
        torch.save(payload, source_dir / 'model_staged.pt')

    def train(checkpoint, iteration, updates, envs, label, warm, max_nan_resets=0):
        stage(checkpoint, iteration)
        text = command(['train', TASK, '--env.scene.num-envs', str(envs), '--agent.max-iterations', str(updates),
            '--agent.save-interval', '100', '--agent.seed', '42', '--agent.run-name', label,
            '--agent.resume', 'True', '--agent.load-run', source_dir.name,
            '--agent.load-checkpoint', 'model_staged.pt', '--enable-nan-guard', 'True'], label, warm=warm,
            max_nan_resets=max_nan_resets)
        assert 'Loading model checkpoint from:' in text
        if warm:
            assert 'weights/normalizer/optimizer kept' in text
        else:
            # Direct check that the schedules resumed: the counter the training
            # process kept must be the staged one, at or past the final stage.
            counter = resumed_counter(text)
            if counter != iteration * STEPS_PER_UPDATE or counter < final_step:
                raise RuntimeError(f'{label} resumed at step counter {counter}, expected '
                                   f'{iteration * STEPS_PER_UPDATE} (final schedule step {final_step})')
        # The last "[course] Actual sampled height counts" line must fit the stage.
        lines = re.findall(r'\[course\] Actual sampled height counts[^:]*: (\[[^\]]*\])', text)
        bands = sampled_bands(json.loads(lines[-1]), state.upper_mm / 1000) if lines else None
        if bands and bands['above_stage']:
            raise RuntimeError(f'{label} sampled heights above the current stage: {bands}')
        directory = sorted(ROOT.glob(f'*_{label}'))[-1]
        return max(directory.glob('model_*.pt'), key=lambda q: int(q.stem.split('_')[-1])), bands

    selection = dict(best=None, best_upper_mm=None, mastered={}, last=None,
                     upper_mm=state.upper_mm, blocks_at_stage=state.blocks_at_stage)
    sampling = {}

    def save():
        state.save(args.output / 'curriculum.json')
        selection.update(upper_mm=state.upper_mm, blocks_at_stage=state.blocks_at_stage)
        (args.output / 'selection.json').write_text(json.dumps(selection, indent=2))

    def quality(block, checkpoint):
        """Walking-quality numbers for one block, from its log and frontier traces."""
        from summarize_height_curriculum import action_std, checkpoint_action_std, walking_quality
        log = args.output / f'{args.label}-b{block:02}.log'
        noise = action_std(log.read_text()) if block else checkpoint_action_std(checkpoint)
        return dict(action_std=noise, **walking_quality(last_frontier['directory']))

    def record(entry, checkpoint, passed_stage):
        entry['sampled_bands'] = sampling.get(entry['block'])
        entry['quality'] = quality(entry['block'], checkpoint)
        selection['last'] = str(checkpoint)
        if passed_stage is not None:
            selection.update(best=str(checkpoint), best_upper_mm=passed_stage)
            selection['mastered'][f'{passed_stage:g}'] = dict(checkpoint=str(checkpoint), sha256=sha(checkpoint))
        save()
        with (args.output / 'stages.jsonl').open('a') as f:
            f.write(json.dumps(entry) + '\n')
        print(f"[height-curriculum] block {entry['block']} at {entry['upper_mm']:g} mm: "
              f"advanced={entry['advanced']} next={entry['next_upper_mm']:g} mm "
              f"would_advance={entry['would_advance']} dev={entry['development_passes']} "
              f"confirm={entry['confirmation_passes']} quality={entry['quality']}", flush=True)

    previous = dict(checkpoint=copied, iteration=source_iteration)

    def train_block(block, warm):
        # The first block warm-starts the source (step counter restarts at 0,
        # weights/normalizer/optimizer kept), unless --resume-source. Later
        # blocks resume the previous block at its cumulative iteration, staged
        # with the current state.
        label = f'{args.label}-b{block:02}'
        checkpoint, bands = train(previous['checkpoint'], previous['iteration'], args.block_updates, 2048,
                                  label, warm, args.max_nan_resets)
        sampling[block] = bands
        (args.output / f'{label}.json').write_text(json.dumps(dict(
            checkpoint=str(checkpoint), sha256=sha(checkpoint), upper_mm=state.upper_mm, sampled_bands=bands)))
        previous.update(checkpoint=checkpoint, iteration=previous['iteration'] + args.block_updates)
        return checkpoint

    try:
        command(['--with', 'pytest', 'pytest', 'tests/test_cliff_course.py', 'tests/test_nan_forensics.py', '-q'],
                'runtime-tests')
        smoke, _ = train(copied, source_iteration, 5, 64, f'{args.label}-smoke', warm=warm_source)
        command(['python', 'scripts/export.py', TASK, '--checkpoint-file', str(smoke),
                 '--num-envs', '1', '--onnx-file', str(args.output / 'smoke.onnx')], 'smoke-export')
        import numpy as np
        import onnx
        import onnxruntime as ort
        graph = onnx.load(str(args.output / 'smoke.onnx')).graph
        assert any(node.op_type == 'Sub' for node in graph.node), 'Missing normalizer centering'
        assert any(node.op_type in ('Div', 'Mul') for node in graph.node), 'Missing normalizer scaling'
        model = ort.InferenceSession(str(args.output / 'smoke.onnx'), providers=['CPUExecutionProvider'])
        output = model.run(None, {model.get_inputs()[0].name: np.zeros((1, 63), dtype=np.float32)})[0]
        assert output.shape == (1, 14) and np.isfinite(output).all()
        save()
        # Baseline: the source under the current rules, with the same check.
        # It does not count toward regression streaks or the stage.
        baseline = check(copied, number=0)
        entry = dict(block=0, checkpoint=str(copied), upper_mm=state.upper_mm, blocks_at_stage=state.blocks_at_stage,
                     development={k: asdict(v) for k, v in baseline.items()},
                     development_passes={k: v.passes for k, v in baseline.items()},
                     regression_streaks={k: None for k in state.regression_conditions()},
                     confirmation=None, confirmation_passes=None, advanced=False, mastered_stage=False,
                     hold=args.hold_height, would_advance=None, next_upper_mm=state.upper_mm,
                     baseline=True, at_unix=time.time())
        entry['quality'] = quality(0, copied)
        with (args.output / 'stages.jsonl').open('a') as f:
            f.write(json.dumps(entry) + '\n')
        print(f"[height-curriculum] baseline at {state.upper_mm:g} mm: dev={entry['development_passes']} "
              f"quality={entry['quality']}", flush=True)
        reason = run_blocks(state, train_block=train_block, check=check, confirm=confirm, now=time.time,
                            deadline=deadline, record=record, max_blocks_per_stage=args.max_blocks_per_stage,
                            hold=args.hold_height, max_blocks=args.blocks if args.hold_height else None,
                            warm_first=warm_source)
        if reason == 'time budget':
            (args.output / 'provisional.json').write_text(json.dumps(dict(
                checkpoint=selection['last'], reason='Not enough time for another block plus checks',
                at_unix=time.time())))
        (args.output / 'complete.json').write_text(json.dumps(dict(
            reason=reason, upper_mm=state.upper_mm, best=selection['best'], last=selection['last'],
            finished_unix=time.time())))
    except TimeoutError as error:
        (args.output / 'provisional.json').write_text(json.dumps(dict(
            checkpoint=selection['last'], reason=repr(error), at_unix=time.time())))
        raise
    except Exception as error:
        (args.output / 'stopped.json').write_text(json.dumps(dict(reason=repr(error), at_unix=time.time())))
        raise
    finally:
        save()
        if NAN_DUMPS.exists():
            shutil.copytree(NAN_DUMPS, args.output / 'nan-dumps', symlinks=True, dirs_exist_ok=True)
        files = [q for q in args.output.rglob('*') if q.is_file() and not q.is_symlink()]
        files += [q for d in ROOT.glob(f'*{args.label}*') for q in d.rglob('*') if q.is_file()]
        (args.output / 'SHA256SUMS').write_text(''.join(
            f'{sha(q)}  {q}\n' for q in files if q.name != 'SHA256SUMS'))


if __name__ == '__main__':
    main()
