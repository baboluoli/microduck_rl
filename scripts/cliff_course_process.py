"""Watch a training process continuously; a NaN reset must end the session.

MuJoCo's NaN guard can dump/reset one environment and let training continue.
The run plan requires stopping instead, even when losses remain finite.
"""
import re
import subprocess
import time
from pathlib import Path


def wait_checked(process: subprocess.Popen, log_path: Path, deadline: float) -> int:
    """Return exit status, or raise promptly on invalid state/loss or deadline.

    The caller owns process-group cleanup in its finally block. Reading the log
    incrementally avoids waiting for a whole 100-update block before noticing a
    guard event. Keep a short tail to recognize messages split between reads.
    """
    tail = ''
    with log_path.open() as log:
        while True:
            tail += log.read()
            if re.search(r'\[NanGuard\] (?:Detected NaN/Inf|NaN/Inf found)', tail):
                raise RuntimeError('Nonfinite simulation state; stop the session')
            if re.search(r'(?i)(loss|reward)[^\n]{0,80}\b(?:nan|inf)\b', tail):
                raise RuntimeError('Nonfinite losses or rewards; stop the session')
            for value in re.findall(r'Episode_Termination/nan_state:\s*([\d.eE+-]+)', tail):
                if float(value) > 0:
                    raise RuntimeError('Nonfinite state termination; stop the session')
            code = process.poll()
            if code is not None:
                # Drain once after exit, so the final write is checked too.
                final = log.read()
                if final:
                    tail += final
                    continue
                return code
            if time.time() >= deadline:
                raise TimeoutError('Retrieval reserve reached')
            tail = tail[-512:]
            time.sleep(.25)
