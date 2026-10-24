"""Bounded single-seed continuation on an already approved, preflighted pod.

Does not provision a pod. Sources must be supplied in chronological order
(original 500, continuation 100, continuation 249). Never modifies source files.
The external watchdog must stop the pod even if this process or SSH dies.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import torch
from mjlab_microduck.tasks.cliff.course_curriculum import Curriculum, Result, rank

TASK = 'Mjlab-CliffOracleCourse-MicroDuck'
ROOT = Path('logs/rsl_rl/oracle_course')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',nargs=3,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True,help='Absolute approved session deadline, including retrieval')
    args = p.parse_args()
    # Reserve 15 minutes for retrieval/shutdown inside the two-hour session.
    deadline = min(args.deadline_epoch, time.time()+7200)-900
    args.output.mkdir(parents=True,exist_ok=False)
    manifest = dict(sources={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in args.sources},
                    deadline_epoch=args.deadline_epoch, training_seed=42, max_updates=1000,
                    training_envs=2048, retrieval_reserve_s=900)
    (args.output/'session.json').write_text(json.dumps(manifest,indent=2))
    import tarfile
    with tarfile.open(args.output/'source.tar.gz','w:gz') as archive:
        for directory in ('src','scripts','tests'):
            for path in Path(directory).rglob('*.py'):
                archive.add(path)
        for path in ('pyproject.toml','uv.lock','AGENTS.md'):
            archive.add(path)
    (args.output/'source.diff').write_bytes(subprocess.check_output(['git','diff','HEAD']))
    environment = dict(os.environ,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',WANDB_MODE='offline')
    def command(command,label,warm=False):
        if time.time() >= deadline: raise TimeoutError('Retrieval reserve reached')
        with (args.output/f'{label}.log').open('w') as log:
            process = subprocess.Popen(['uv','run',*command],stdout=log,stderr=subprocess.STDOUT,
                env=dict(environment,MICRODUCK_WARM_START='1' if warm else '0'),start_new_session=True)
            try:
                code = process.wait(timeout=max(1,deadline-time.time()))
                if code: raise RuntimeError(f'{label} exit {code}')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
        text = (args.output/f'{label}.log').read_text()
        import re
        if re.search(r'(?i)(loss|reward)[^\n]{0,80}\b(?:nan|inf)\b',text):
            raise RuntimeError('Nonfinite losses or rewards')
        if '[NanGuard] NaN/Inf found' in text: raise RuntimeError('Nonfinite training state')
        return text

    count = 0
    def evaluate(checkpoint,upper,seeds,distances=(.45,.55,.65),videos=False):
        nonlocal count
        count += 1
        directory = args.output/f'eval-{count:03}'
        heights = [(.02+i*.0025) for i in range(round((upper-.02)/.0025)+1)]
        command(['python','scripts/eval_cliff_course.py',str(checkpoint),'--output',str(directory),
                 '--heights',*map(str,heights),'--distances',*map(str,distances),
                 '--seeds',*map(str,seeds),*(['--videos'] if videos else [])],f'eval-{count:03}')
        batch = {}
        for h in heights:
            for d in distances:
                groups = [json.loads((directory/f'h{h:.4f}-d{d:.2f}-s{s}.json').read_text()) for s in seeds]
                assert all(len(g)==24 for g in groups)
                if any('nonfinite' in r['events'] for g in groups for r in g):
                    raise RuntimeError('Nonfinite evaluation state')
                batch[f'{h*1000:g}/{round(d*100)}'] = Result(
                    tuple(sum(r['success'] for r in g) for g in groups),
                    sum('impact' in r['events'] for g in groups for r in g),
                    sum(r['one_metre'] for g in groups for r in g))
        return batch

    command(['--with','pytest','pytest','tests/test_cliff_course.py','-q'],'runtime-tests')
    # Fully evaluate every candidate before selection. Strict > preserves ties.
    candidates = [(source,evaluate(source,.02,(20,21,22),(.55,),True)) for source in args.sources]
    source,best_batch = candidates[0]
    for checkpoint,batch in candidates[1:]:
        if rank(batch.values()) > rank(best_batch.values()): source,best_batch=checkpoint,batch
    evaluate(source,.02,(40,41,42),(.55,))
    development = evaluate(source,.02,(20,21,22))
    confirmation = evaluate(source,.02,(40,41,42))
    state = Curriculum()
    # A healthy timeout close to the finish is evidence to review the deadline,
    # not permission to train a faster gait under an accidental speed floor.
    state.advance(development,confirmation,str(source))
    # Baseline audit is emitted by evaluator. Main always restarts from the
    # unchanged selected source; smoke weights are never used for the pilot.
    ROOT.mkdir(parents=True,exist_ok=True)
    source_dir = ROOT/'course-session-source'
    source_dir.mkdir(exist_ok=False)
    copied_source = source_dir/'model_source.pt'
    shutil.copy2(source,copied_source)
    def staged_checkpoint(checkpoint,iteration):
        payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
        payload['iter'] = iteration
        payload.setdefault('infos',{})
        if payload['infos'] is None: payload['infos'] = {}
        payload['infos']['course_curriculum'] = asdict(state)
        payload['infos']['env_state'] = {'common_step_counter':iteration*24}
        torch.save(payload,source_dir/'model_staged.pt')
    def train(checkpoint,iteration,limit,n,label,warm=False):
        staged_checkpoint(checkpoint,iteration)
        text = command(['train',TASK,'--env.scene.num-envs',str(n),'--agent.max-iterations',str(limit-iteration),
            '--agent.save-interval','100','--agent.seed','42','--agent.run-name',label,
            '--agent.resume','True','--agent.load-run','course-session-source',
            '--agent.load-checkpoint','model_staged.pt','--enable-nan-guard','True'],label,warm)
        assert 'Loading model checkpoint from:' in text
        if warm: assert 'weights/normalizer/optimizer kept' in text
        directory = sorted(ROOT.glob(f'*_{label}'))[-1]
        return max(directory.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    smoke = train(copied_source,0,5,64,'course-smoke',True)
    command(['python','scripts/export.py',TASK,'--checkpoint-file',str(smoke),
             '--num-envs','1','--onnx-file',str(args.output/'smoke.onnx')],'smoke-export')
    import onnxruntime as ort
    import numpy as np
    import onnx
    graph = onnx.load(str(args.output/'smoke.onnx')).graph
    assert any(node.op_type == 'Sub' for node in graph.node), 'Missing normalizer centering'
    assert any(node.op_type in ('Div','Mul') for node in graph.node), 'Missing normalizer scaling'
    model = ort.InferenceSession(str(args.output/'smoke.onnx'),providers=['CPUExecutionProvider'])
    output = model.run(None,{model.get_inputs()[0].name:np.zeros((1,63),dtype=np.float32)})[0]
    assert output.shape == (1,14) and np.isfinite(output).all()
    current = copied_source
    best = source
    entering = evaluate(source,state.upper_mm/1000,(20,21,22))
    best_rank = rank([r for k,r in entering.items() if k.startswith(f'{state.upper_mm:g}/')])
    best_upper = state.upper_mm/1000
    last_check = 0
    try:
        for iteration in range(0,1000,100):
            current = train(current,iteration,iteration+100,2048,f'course-{iteration+100}',iteration==0)
            # Behavior checks start at least seven minutes apart. Never wait
            # into retrieval reserve or start an extra stage to consume time.
            delay = max(0,420-(time.time()-last_check))
            if time.time()+delay >= deadline:
                (args.output/'provisional.json').write_text(json.dumps(dict(checkpoint=str(current),reason='Cannot complete final check within spacing/deadline')))
                break
            while delay > 0:
                time.sleep(min(30,delay))
                delay = max(0,420-(time.time()-last_check))
            last_check = time.time()
            upper = state.upper_mm/1000
            batch = evaluate(current,upper,(20,21,22))
            if state.regression(batch): break
            mastered = all(batch[k].passes for k in state.baselines)
            frontier = [r for k,r in batch.items() if k.startswith(f'{state.upper_mm:g}/')]
            score = rank(frontier)
            if mastered and (best_rank is None or score > best_rank):
                best,best_rank=current,score
                best_upper=upper
            if all(r.passes for r in batch.values()):
                confirmed = evaluate(current,upper,(40,41,42))
                if state.advance(batch,confirmed,str(current)):
                    best,best_rank=current,None
                    best_upper=upper
                    # Measure the entering policy at the new endpoint before
                    # training there. Failure holds the admitted frontier.
                    entering = evaluate(current,state.upper_mm/1000,(20,21,22))
                    best_rank = rank([r for k,r in entering.items() if k.startswith(f'{state.upper_mm:g}/')])
                    best_upper = state.upper_mm/1000
            state.save(args.output/'curriculum.json')
            (args.output/'selection.json').write_text(json.dumps(dict(best=str(best),last=str(current),upper_mm=state.upper_mm)))
        # Held-out batches are used once for final review, never selection.
        if best != source:
            evaluate(best,best_upper,(30,31,32),videos=True)
    except TimeoutError:
        (args.output/'provisional.json').write_text(json.dumps(dict(checkpoint=str(current),reason='Session work deadline reached; no unevaluated checkpoint promoted')))
    finally:
        state.save(args.output/'curriculum.json')
        # A checksum list allows retrieval verification; it does not shut down
        # the pod. An independent external watchdog is required by the run plan.
        paths = list(args.output.rglob('*'))+[p for directory in ROOT.glob('*course*') for p in directory.rglob('*') if p.is_file()]
        (args.output/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n' for p in paths if p.is_file() and p.name!='SHA256SUMS'))


if __name__ == '__main__':
    main()
