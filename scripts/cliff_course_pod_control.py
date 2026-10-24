"""Local watchdog/retrieval helpers for an explicitly approved RunPod session.

This process depends on the Mac remaining awake and online. It never deletes a
pod or volume. Run watchdog before starting remote work, and retire it only after
retrieval and independently confirmed stopped status.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['watch','retrieve','stop'])
    p.add_argument('--pod-id',required=True)
    p.add_argument('--deadline-epoch',type=float)
    p.add_argument('--ssh-host')
    p.add_argument('--remote-root')
    p.add_argument('--output-relative',default='logs/course-session')
    p.add_argument('--destination',type=Path)
    args = p.parse_args()
    def stop():
        subprocess.run(['uv','run','--no-sync','runpodctl','stop','pod',args.pod_id],check=True)
        for _ in range(12):
            result = subprocess.run(['uv','run','--no-sync','runpodctl','get','pod',args.pod_id],
                                    check=True,capture_output=True,text=True)
            state = json.loads(result.stdout)
            records = state if isinstance(state,list) else [state]
            if any(str(row.get('desiredStatus',row.get('status',''))).upper() in ('EXITED','STOPPED') for row in records):
                print(f'Confirmed stopped status for {args.pod_id}')
                return
            time.sleep(5)
        raise RuntimeError('Stop requested, but stopped status was not confirmed; inspect pod immediately')
    if args.action == 'stop':
        stop()
    elif args.action == 'watch':
        if args.deadline_epoch is None: p.error('--deadline-epoch required')
        while time.time() < args.deadline_epoch:
            time.sleep(min(30,args.deadline_epoch-time.time()))
        stop()
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
        subprocess.run(['scp',f'{args.ssh_host}:{shlex.quote(remote+args.output_relative+"/SHA256SUMS")}',str(manifest)],check=True)
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
        subprocess.run(['rsync','-a','--protect-args','--files-from',str(file_list),
                        f'{args.ssh_host}:{remote}',str(args.destination)+'/'],check=True)
        for path,digest in expected.items():
            actual=hashlib.sha256((args.destination/path).read_bytes()).hexdigest()
            if actual!=digest: raise RuntimeError(f'Checksum mismatch: {path}')
        print(f'Verified {len(paths)} artifacts; run stop and verify stopped status next.')


if __name__ == '__main__': main()
