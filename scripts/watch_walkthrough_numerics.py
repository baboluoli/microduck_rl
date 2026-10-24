"""Keep the numerical stop active while the session runner evaluates a policy.

This process reads a log, not simulator state, and never changes training
parameters. Killing the training process group stops both uv and its child.
Run only against the process ID created for this session's training run.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import time

PATTERN = re.compile(r'(?i)(loss|reward|entropy|std)[^\n]{0,80}\b(?:nan|inf)\b')


def nonfinite(text):
    return '[NanGuard] NaN/Inf found' in text or bool(PATTERN.search(text))


def watch(pid, log, result):
    group = os.getpgid(pid)
    assert group != os.getpgrp(), 'Refuse to monitor our own process group'
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        if log.exists() and nonfinite(log.read_text()):
            result.write_text(json.dumps(dict(stop='Nonfinite training state/loss', pid=pid,
                                              process_group=group, time=time.time()), indent=2))
            os.killpg(group, signal.SIGTERM)
            return
        time.sleep(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pid', type=int)
    p.add_argument('log', type=Path)
    p.add_argument('result', type=Path)
    a = p.parse_args()
    watch(a.pid, a.log, a.result)
