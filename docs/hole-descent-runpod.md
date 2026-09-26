# Hole descent task: first training pass

This task trains the Microduck **legs** policy in the existing `microduck_rl`
PPO/MuJoCo-Warp stack. Every generated terrain tile samples a flat or gently
sloping approach, a shallow ramped trench, and a flat bottom. PPO supplies the
joint targets; the environment supplies only a forward velocity command. The
observation/action ABI remains the existing 61 → 14 contract.

## Run it on RunPod

The MacBook is for editing and reviewing. Training should run in a Linux GPU
pod with a CUDA-capable NVIDIA GPU. In your RunPod pod, clone the fork and
task branch that contain this environment, then run:

```bash
cd /workspace
git clone --branch codex/hole-descent https://github.com/baboluoli/microduck_rl.git
cd microduck_rl
uv sync

# Verify that the task is registered.
uv run list-envs | rg HoleDescent

# Start with a short, inexpensive run to check the pod and task wiring.
uv run train Mjlab-HoleDescent-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 5 \
  --agent.run-name hole-descent-smoke

# Train on the GPU. Keep logs/checkpoints on a mounted persistent volume.
uv run train Mjlab-HoleDescent-Flat-MicroDuck \
  --env.scene.num-envs 2048 \
  --agent.max_iterations 10000 \
  --agent.run-name hole-descent
```

Choose a pod with enough GPU memory for the requested number of parallel
environments. If the pod runs out of memory, lower `--env.scene.num-envs` (for
example, from 2048 to 1024). The official training workflow exports ONNX with
`uv run scripts/export.py` after selecting a checkpoint; do not hand-convert a
checkpoint because export bakes the observation normalizer into the policy.

## Current scope

This first task teaches a single forward walking policy to handle randomized
flat/slope approaches and trench geometry. It does not yet search for a hole
at an arbitrary location or choose between multiple routes. Those need a
goal-conditioned navigation policy and terrain observations, which would be a
separate policy/runtime interface.
