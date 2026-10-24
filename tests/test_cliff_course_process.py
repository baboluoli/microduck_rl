"""A real subprocess proves invalid state is caught before the block exits."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import time
import pytest

spec = importlib.util.spec_from_file_location('cliff_course_process', Path(__file__).parents[1]/'scripts/cliff_course_process.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('message', ['[NanGuard] Detected NaN/Inf at step 7544',
    '[NanGuard] NaN/Inf found in envs: [1232]', 'Mean value loss: nan',
    'Episode_Termination/nan_state: 0.0417'])
def test_invalid_state_interrupts_running_process(tmp_path, message):
    path=tmp_path/'child.log'
    with path.open('w') as output:
        process=subprocess.Popen([sys.executable,'-c',
            'import time; print('+repr(message)+', flush=True); time.sleep(30)'], stdout=output)
        started=time.time()
        try:
            with pytest.raises(RuntimeError):
                module.wait_checked(process,path,time.time()+5)
            assert process.poll() is None, 'Must detect the event before child completion'
            assert time.time()-started < 3
        finally:
            process.terminate(); process.wait(timeout=5)


def test_finite_child_exits_normally(tmp_path):
    path=tmp_path/'child.log'
    with path.open('w') as output:
        process=subprocess.Popen([sys.executable,'-c',
            'print("Mean reward: 12.3\\nEpisode_Termination/nan_state: 0.0000")'],stdout=output)
        assert module.wait_checked(process,path,time.time()+5)==0
