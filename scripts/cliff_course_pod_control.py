"""Local watchdog/retrieval helpers for an explicitly approved RunPod or Vast.ai session.

This process depends on the Mac remaining awake and online. It never deletes a
pod, instance or volume (destroying a Vast instance stays a separate, explicit
step after retrieval). Run watchdog before starting remote work, and retire it only after
retrieval and independently confirmed stopped status.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


# How each provider stops an instance and reports its state. Vast's
# ``actual_status`` reads ``exited`` once a stopped container is down.
STOPPED = {'runpod': {'STOPPED'}, 'vast': {'STOPPED', 'EXITED'}}


def stop_command(provider, pod_id):
    if provider == 'vast':
        return ['vastai', 'stop', 'instance', pod_id]
    return ['uv', 'run', '--no-sync', 'runpodctl', 'pod', 'stop', pod_id]


def status_command(provider, pod_id):
    if provider == 'vast':
        return ['vastai', 'show', 'instance', pod_id, '--raw']
    return ['uv', 'run', '--no-sync', 'runpodctl', 'pod', 'get', pod_id]


def is_stopped(provider, stdout):
    state = json.loads(stdout)
    records = state if isinstance(state, list) else [state]
    field = 'actual_status' if provider == 'vast' else 'runtimeStatus'
    return any(str(row.get(field) or '').upper() in STOPPED[provider] for row in records)


def stop(provider, pod_id, attempts=12, delay=5):
    subprocess.run(stop_command(provider, pod_id), check=True)
    for _ in range(attempts):
        result = subprocess.run(status_command(provider, pod_id), check=True, capture_output=True, text=True)
        if is_stopped(provider, result.stdout):
            print(f'Confirmed stopped status for {pod_id}')
            return
        time.sleep(delay)
    raise RuntimeError('Stop requested, but stopped status was not confirmed; inspect pod immediately')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['watch','retrieve','stop'])
    p.add_argument('--pod-id',required=True,help='RunPod pod id or Vast instance id')
    p.add_argument('--provider',choices=['runpod','vast'],default='runpod')
    p.add_argument('--deadline-epoch',type=float)
    p.add_argument('--ssh-host')
    p.add_argument('--ssh-config',type=Path,help='Session-specific OpenSSH config (including port and key)')
    p.add_argument('--remote-root')
    p.add_argument('--output-relative',default='logs/course-session')
    p.add_argument('--destination',type=Path)
    args = p.parse_args()
    if args.action == 'stop':
        stop(args.provider,args.pod_id)
    elif args.action == 'watch':
        if args.deadline_epoch is None: p.error('--deadline-epoch required')
        while time.time() < args.deadline_epoch:
            time.sleep(max(0,min(30,args.deadline_epoch-time.time())))
        stop(args.provider,args.pod_id)
    else:
        if not all((args.ssh_host,args.remote_root,args.destination)):
            p.error('retrieve requires --ssh-host, --remote-root and --destination')
        if Path(args.output_relative).is_absolute() or '..' in Path(args.output_relative).parts:
            p.error('output must be relative to project')
        args.destination.mkdir(parents=True,exist_ok=True)
        remote = args.remote_root.rstrip('/')+'/'
        # Paths in SHA256SUMS are relative to the remote project. Fetch that
        # structure so local validation can resolve the identical paths.
        manifest = args.destination/args.output_relative/'SHA256SUMS'
        manifest.parent.mkdir(parents=True,exist_ok=True)
        ssh_options = ['-F',str(args.ssh_config)] if args.ssh_config else []
        subprocess.run(['scp',*ssh_options,f'{args.ssh_host}:{shlex.quote(remote+args.output_relative+"/SHA256SUMS")}',str(manifest)],check=True)
        paths = []
        expected = {}
        for line in manifest.read_text().splitlines():
            digest,path = line.split('  ',1)
            if Path(path).is_absolute() or '..' in Path(path).parts:
                raise ValueError('Unsafe manifest path')
            paths.append(path)
            expected[path]=digest
        file_list = args.destination/'retrieval-files.txt'
        file_list.write_text('\n'.join(paths)+'\n')
        subprocess.run(['rsync','-a','-e',shlex.join(['ssh',*ssh_options]),'--files-from',str(file_list),
                        f'{args.ssh_host}:{remote}',str(args.destination)+'/'],check=True)
        for path,digest in expected.items():
            actual=hashlib.sha256((args.destination/path).read_bytes()).hexdigest()
            if actual!=digest: raise RuntimeError(f'Checksum mismatch: {path}')
        print(f'Verified {len(paths)} artifacts; run stop and verify stopped status next.')


if __name__ == '__main__': main()
