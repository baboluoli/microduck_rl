# Cliff descent

Read these files in order:

1. `env_cfg.py`: builds on the shared walking environment, then sets cliff
   terrain, commands, contact sensors, rewards, and episode endings. Contains
   the standard 61-input task and the separate 63-input oracle experiment.
2. `mdp.py`: cliff observations, midair resets, forward progress, impact
   disqualification, and landing success. MDP means Markov Decision Process.
3. `terrain.py`: constructs the upper platform, ledge, and lower floor.

Shared walking behavior stays in `../microduck_velocity_env_cfg.py` and
`../mdp.py`. Task registration stays in `../__init__.py`. The old cliff
configuration and terrain modules are compatibility imports for saved configs.

From the RL project root:

- `scripts/eval_cliff_drop.py`: checkpoint evaluation and video recording.
- `scripts/run_cliff_steering_pilot.py`: baseline evaluation, then evaluation
  every 25 PPO updates, and a final battery; also checks flat-ground walking.
- `scripts/run_cliff_checkpoint_pilot.py`: older pilot; evaluates the baseline
  and after 100, 200, and 250 updates, unless stopped early.
- `scripts/prepare_cliff_steering_checkpoint.py`: prepares the original
  zero-yaw checkpoint for steering training.
- `docs/cliff-steering-pilot-report.md`: trained 2 cm results and artifacts.
- `docs/cliff-oracle-2cm.md`: exact-terrain-input experiment and its status.

Ordinary `uv run train` does not automatically run the separate evaluator.
Runpod training requires approval before starting.
