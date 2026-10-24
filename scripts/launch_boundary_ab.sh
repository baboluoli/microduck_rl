#!/bin/bash
# Run the boundary-semantics A/B session on an already created RunPod pod.
#
# Usage (from tinyduck/lemming/rl, on the Mac):
#   scripts/launch_boundary_ab.sh POD_ID SSH_HOST SSH_PORT
#
# Create the pod first, as for the 2026-10-24 course session: A40 48 GB,
# image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404, SSH exposed, your
# runpodctl key at ~/.runpod/ssh/runpodctl-ssh-key.
#
# What it does, in order: local CPU tests; refuse uncommitted code; start a
# local watchdog that stops the pod at the 2-hour deadline; upload code,
# checkpoints and curriculum; bootstrap + EGL/CUDA preflight; run
# scripts/run_boundary_ab_session.py; poll until it ends; retrieve and verify
# checksums; stop the pod and confirm STOPPED. The pod is stopped on any exit.
# Expected cost at $0.59/h: about $0.6-1.2. Never deletes a pod or volume.
set -euo pipefail
cd "$(dirname "$0")/.."
POD=${1:?POD_ID}; HOST=${2:?SSH_HOST}; PORT=${3:?SSH_PORT}
S=logs/boundary-ab-20261024
OLD=logs/course-session-20261024/artifacts
SOURCE=$OLD/logs/rsl_rl/oracle_course/course-session-source/model_source.pt
MODEL99=$OLD/logs/rsl_rl/oracle_course/2026-10-24_18-58-57_course-100/model_99.pt
CURRICULUM=$OLD/logs/course-session/curriculum.json
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
mkdir -p "$S"

echo "== Local CPU tests"
uv run --with pytest pytest tests/test_nan_forensics.py tests/test_cliff_course.py tests/test_cliff_course_process.py -q
if [ -n "$(git status --porcelain -- src scripts tests pyproject.toml uv.lock)" ]; then
  echo "Uncommitted code changes; commit first so the run is reproducible." >&2; exit 1
fi
git rev-parse HEAD > "$S/COMMIT"

DEADLINE=$(( $(date +%s) + 7200 ))
echo "$DEADLINE" > "$S/deadline_epoch"
ssh-keyscan -p "$PORT" "$HOST" > "$S/known_hosts" 2>/dev/null
cat > "$S/ssh_config" <<EOF
Host boundary-ab
 HostName $HOST
 Port $PORT
 User root
 IdentityFile $KEY
 UserKnownHostsFile $PWD/$S/known_hosts
 StrictHostKeyChecking yes
 ConnectTimeout 15
 ServerAliveInterval 15
 ServerAliveCountMax 3
EOF
SSH=(ssh -F "$S/ssh_config" boundary-ab)

echo "== Watchdog (stops pod at $(date -r "$DEADLINE"))"
nohup uv run --no-sync python scripts/cliff_course_pod_control.py watch --pod-id "$POD" \
  --deadline-epoch "$DEADLINE" > "$S/watchdog.log" 2>&1 &
WATCHDOG=$!
echo "$WATCHDOG" > "$S/watchdog.pid"
finish() {
  echo "== Stopping pod $POD"
  uv run --no-sync python scripts/cliff_course_pod_control.py stop --pod-id "$POD" | tee "$S/stop.log" || \
    echo "!! STOP NOT CONFIRMED - stop pod $POD manually now" >&2
  kill "$WATCHDOG" 2>/dev/null || true
}
trap finish EXIT

echo "== Upload"
git ls-files -co --exclude-standard . | grep -v '^logs/' | tar -czf "$S/source.tar.gz" -T -
"${SSH[@]}" 'mkdir -p /workspace/rl/sources'
scp -F "$S/ssh_config" "$S/source.tar.gz" boundary-ab:/workspace/boundary-ab-source.tar.gz
scp -F "$S/ssh_config" "$SOURCE" boundary-ab:/workspace/rl/sources/model_source.pt
scp -F "$S/ssh_config" "$MODEL99" boundary-ab:/workspace/rl/sources/model_99.pt
scp -F "$S/ssh_config" "$CURRICULUM" boundary-ab:/workspace/rl/sources/curriculum.json
scp -F "$S/ssh_config" "$S/COMMIT" boundary-ab:/workspace/rl/sources/COMMIT

echo "== Bootstrap and start session"
"${SSH[@]}" "cat > /workspace/boundary-ab.sh" <<EOF
#!/bin/bash
set -euo pipefail
cd /workspace/rl
tar --no-same-owner -xzf /workspace/boundary-ab-source.tar.gz -C /workspace/rl
apt-get update -q
apt-get install -y -q --no-install-recommends libegl1 libgles2 libgl1 libglx-mesa0 libglib2.0-0
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
# No .git on the pod: evaluation manifests read the uploaded commit from here.
export MICRODUCK_GIT_HEAD="\$(cat sources/COMMIT)"
export PATH="/root/.local/bin:\$PATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=offline PYTHONUNBUFFERED=1
uv sync --frozen
uv run python - <<'PREFLIGHT'
import ctypes, torch
ctypes.CDLL("libEGL.so.1"); ctypes.CDLL("libGLESv2.so.2")
assert torch.cuda.is_available(), "CUDA unavailable"
import mujoco
context = mujoco.GLContext(16, 16)
try:
    context.make_current()
    print("CUDA and headless EGL context passed:", torch.cuda.get_device_name(0), flush=True)
finally:
    context.free()
PREFLIGHT
uv run python scripts/run_boundary_ab_session.py --source sources/model_source.pt \
  --old-trained sources/model_99.pt --curriculum sources/curriculum.json \
  --output logs/boundary-ab --deadline-epoch $DEADLINE
EOF
"${SSH[@]}" 'chmod +x /workspace/boundary-ab.sh; rm -f /workspace/boundary-ab-exit;
  nohup bash -c "/workspace/boundary-ab.sh > /workspace/boundary-ab-bootstrap.log 2>&1; echo \$? > /workspace/boundary-ab-exit" >/dev/null 2>&1 &'

echo "== Monitoring (Ctrl-C stops the pod)"
while true; do
  sleep 60
  STATUS=$("${SSH[@]}" 'cat /workspace/boundary-ab-exit 2>/dev/null; echo ---; tail -n 2 /workspace/boundary-ab-bootstrap.log; ls /workspace/rl/logs/boundary-ab 2>/dev/null | grep -E "^(eval|boundary|runtime|smoke).*log$" | tail -n 1' || echo "ssh failed")
  echo "[$(date +%H:%M:%S)] $(echo "$STATUS" | tr '\n' ' ' | cut -c1-200)"
  if [ "$(echo "$STATUS" | head -n 1)" != "---" ] && [ "$(echo "$STATUS" | head -n 1)" != "ssh failed" ]; then break; fi
  if [ "$(date +%s)" -ge $((DEADLINE - 600)) ]; then echo "Deadline reserve reached"; break; fi
done

echo "== Retrieve"
scp -F "$S/ssh_config" boundary-ab:/workspace/boundary-ab-bootstrap.log "$S/" || true
uv run --no-sync python scripts/cliff_course_pod_control.py retrieve --pod-id "$POD" --ssh-host boundary-ab \
  --ssh-config "$S/ssh_config" --remote-root /workspace/rl --output-relative logs/boundary-ab \
  --destination "$S/artifacts" || echo "!! Retrieval failed; the pod volume is kept (stopped, not deleted)." >&2
cat "$S/artifacts/logs/boundary-ab/summary.json" 2>/dev/null || true
