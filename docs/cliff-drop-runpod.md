# Cliff drop: first training pass

The first 3,000-update pilot did not learn the variable-height objective. Read
the [run report](cliff-drop-3000-report.md) before repeating or extending it.

`Mjlab-CliffDrop-Flat-MicroDuck` starts Microduck 55 cm before a vertical ledge.
The lower landing floor is 2–10 cm below the upper platform. The terrain
curriculum starts at 2 cm and raises the maximum sampled drop height only
after successful episodes. There is no ramp, route search, or steering task.
The actor keeps the standard 61-observation / 14-action runtime interface.
During training, 35% of resets begin just past the edge and already falling,
so the policy can practice landings before it learns the entire approach.
Play mode always begins on the upper platform.

An episode succeeds when the robot travels at least 40 cm beyond the edge,
has a foot on the lower floor, is within 30° of upright, has settled vertical
motion, and walks forward for 0.3 s. A head contact above 15 N or trunk
contact above 20 N invalidates that episode's success. Servo housing impact,
servo acceleration spikes, stalls, and head/trunk impact costs shape the
landing. These force limits are simulation
training criteria, not a claim of hardware safety; inspect landing videos and
force traces before trying a drop on the real robot.

## RunPod training

Use a Linux CUDA pod with persistent storage for logs and checkpoints. First
run the short 64-environment smoke run, then stop at 1,000 iterations to
inspect a checkpoint before spending time on a longer run:

```bash
uv run list-envs | rg CliffDrop
uv run train Mjlab-CliffDrop-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5 \
  --agent.run-name cliff-drop-smoke

uv run train Mjlab-CliffDrop-Flat-MicroDuck \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 1000 \
  --agent.run-name cliff-drop-first-1000
```

The runner saves `model_250.pt`, `model_500.pt`, `model_750.pt`, and
`model_999.pt` (`save_interval=250`; the final checkpoint uses the zero-based
last iteration). On a GPU pod, inspect the selected
checkpoint in the simulator:

```bash
uv run play Mjlab-CliffDrop-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/cliff_drop/<run-directory>/model_999.pt
```

Resume from that checkpoint with `--agent.resume True` and
`--agent.load-checkpoint model_999.pt`. In mjlab 1.3.0,
`--agent.max-iterations` counts **additional** iterations, so 9,000 more
reaches 10,000 total:

```bash
uv run train Mjlab-CliffDrop-Flat-MicroDuck \
  --env.scene.num-envs 2048 \
  --agent.resume True \
  --agent.load-run <run-directory> \
  --agent.load-checkpoint model_999.pt \
  --agent.max-iterations 9000 \
  --agent.run-name cliff-drop-continued
```

Keep the first run's checkpoint accessible when resuming. Training state
(policy, optimizer, normalizers, and iteration counter) is restored; W&B may
start a separate logging run. Before training, inspect a generated tile in
simulation to confirm the upper and lower collision floors form a clean edge.
Judge the checkpoint by success rate **at each drop height**, landing contacts,
and whether it actually walks after impact. Export a selected checkpoint with
`uv run scripts/export.py` so the observation normalizer is included.
