"""Watch a training process continuously; NaN resets are counted and capped.

MuJoCo's NaN guard dumps the first broken environment (once per process) and
the ``nan_state`` termination resets every broken environment, so training can
continue safely. ``max_nan_resets=0`` keeps the strict rule: any NaN ends the
session. A positive cap tolerates rare solver glitches but still stops a run
whose NaNs become frequent. Nonfinite losses or rewards always stop it, because
those mean the bad numbers reached the network.
"""
import re
import subprocess
import time
from pathlib import Path


def wait_checked(process: subprocess.Popen, log_path: Path, deadline: float,
                 max_nan_resets: int = 0) -> int:
    """Return exit status, or raise promptly on invalid state/loss or deadline.

    The caller owns process-group cleanup in its finally block. Reading the log
    incrementally avoids waiting for a whole 100-update block before noticing a
    guard event. Only complete lines are inspected, so a message split between
    reads is seen once, whole, and each event is counted exactly once.

    NaN events counted: the guard's first detection, plus every logged
    iteration whose ``Episode_Termination/nan_state`` is non-zero. That value is
    a per-iteration statistic, not an exact env count, so the cap is a
    frequency limit rather than a precise number of broken environments.
    """
    pending = ''
    nan_events = 0
    with log_path.open() as log:
        while True:
            pending += log.read()
            *lines, pending = pending.split('\n')
            for line in lines:
                if re.search(r'(?i)(loss|reward)[^\n]{0,80}\b(?:nan|inf)\b', line):
                    raise RuntimeError('Nonfinite losses or rewards; stop the session')
                event = re.search(r'\[NanGuard\] Detected NaN/Inf', line)
                value = re.search(r'Episode_Termination/nan_state:\s*([\d.eE+-]+)', line)
                if event or (value and float(value.group(1)) > 0):
                    nan_events += 1
                    print(f'[wait_checked] NaN event {nan_events}/{max_nan_resets} allowed: {line.strip()}', flush=True)
                    if nan_events > max_nan_resets:
                        raise RuntimeError('Nonfinite simulation state; stop the session')
            code = process.poll()
            if code is not None:
                # Drain once after exit, so the final write is checked too.
                final = log.read()
                if final:
                    pending += final
                    continue
                if pending:
                    pending += '\n'  # Inspect an unterminated last line.
                    continue
                return code
            if time.time() >= deadline:
                raise TimeoutError('Retrieval reserve reached')
            time.sleep(.25)
