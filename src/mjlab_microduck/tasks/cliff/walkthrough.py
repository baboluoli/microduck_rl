"""Episode state for the fixed-start walk-through experiment.

All tensors have one row per simulated robot. A milestone is a remembered
landing, not a reward. Clean walking blocks earn 8 points each (up to five); strict timeout earns 40.
This separate module leaves the historical oracle's landing rule untouched.
"""
import math
import torch


def lower_floor_contact(found, pos, origins):
    """Check the actual lower box top, in coordinates relative to the spawn.

    The tile runs from -0.95 to 4.05 relative to its spawn, with half-width
    1.35 m. The edge is at 0.55 and the lower top is at -0.02 m.
    """
    p = pos.reshape(len(origins), -1, 3) - origins[:, None, :]
    valid = (found.reshape(len(origins), -1).bool()
             & torch.isfinite(p).all(-1)
             & (p[..., 0] >= .55) & (p[..., 0] <= 4.05)
             & (p[..., 1].abs() <= 1.35)
             & ((p[..., 2] + .02).abs() <= .008))
    return valid.any(-1)


class WalkingState:
    """One update per control step; reset only the selected episode rows."""
    def __init__(self, n, device, dt):
        self.dt = dt
        self.block_steps = round(1. / dt)
        if abs(self.block_steps * dt - 1.) > 1e-6:
            raise ValueError('One-second blocks must contain whole control steps')
        self.last_step = -1
        self.stable = torch.zeros(n, device=device)
        self.milestone_step = torch.full((n,), -1, dtype=torch.long, device=device)
        self.blocks = torch.zeros_like(self.milestone_step)
        self.milestone_x = torch.zeros(n, device=device)
        self.block_x = torch.zeros_like(self.milestone_x)
        self.bad = torch.zeros(n, dtype=torch.bool, device=device)
        self.impact = torch.zeros_like(self.bad)
        self.boundary = torch.zeros_like(self.bad)
        self.nonfinite = torch.zeros_like(self.bad)
        self.paid = torch.zeros_like(self.bad)
        self.pulse = torch.zeros_like(self.bad)
        # These counters never count observed blocks after a latched failure.
        self.clean_blocks = torch.zeros_like(self.blocks)
        self.paid_blocks = torch.zeros_like(self.blocks)
        self.block_pulse = torch.zeros_like(self.bad)

    def reset(self, ids):
        for name in ('stable', 'blocks', 'milestone_x', 'block_x', 'bad',
                     'impact', 'boundary', 'nonfinite', 'paid', 'pulse',
                     'clean_blocks', 'paid_blocks', 'block_pulse'):
            getattr(self, name)[ids] = 0
        self.milestone_step[ids] = -1

    def update(self, *, step, age, x, y, vx, vz, cos_tilt, supported,
               lower, head, trunk, finite, horizon):
        if step == self.last_step:
            return self.pulse
        self.last_step = step
        self.reset(age <= 1)
        self.pulse.zero_()
        self.block_pulse.zero_()
        self.nonfinite |= ~finite
        self.impact |= (head > 15) | (trunk > 20)
        self.boundary |= (x < -.75) | (x > 3.8) | (y.abs() > 1.15)
        # After the milestone, remember even brief tilt violations. Before
        # the milestone the inherited half-second fall termination still applies.
        self.bad |= (self.nonfinite | self.impact | self.boundary
                     | ((self.milestone_step >= 0) & (cos_tilt <= math.cos(math.radians(45)))))
        stable = ((x >= .95) & supported & (cos_tilt >= math.cos(math.radians(30)))
                  & (vz.abs() <= .25) & (vx >= .05) & ~self.impact & finite)
        self.stable = torch.where(stable, self.stable + self.dt, torch.zeros_like(self.stable))
        milestone = (self.milestone_step < 0) & (self.stable >= .30) & lower
        self.milestone_step[milestone] = age[milestone]
        self.milestone_x[milestone] = x[milestone]
        self.block_x[milestone] = x[milestone]
        elapsed = age - self.milestone_step
        complete_block = ((self.milestone_step >= 0) & (elapsed > 0)
                          & (elapsed % self.block_steps == 0))
        self.bad |= complete_block & ((x - self.block_x) < .05 - 1e-6)
        self.blocks += complete_block.long()
        self.clean_blocks += (complete_block & ~self.bad).long()
        self.block_pulse = complete_block & ~self.bad & (self.paid_blocks < 5)
        self.paid_blocks += self.block_pulse.long()
        self.block_x[complete_block] = x[complete_block]
        # Checking age, rather than a manager's timeout buffer, makes the
        # result independent of termination-term ordering.
        self.pulse = ((age == horizon) & (self.milestone_step >= 0)
                      & (self.blocks >= 5) & ~self.bad & ~self.paid)
        self.paid |= self.pulse
        return self.pulse


def update(env):
    """Read simulator evidence once, shared by reward and boundary termination."""
    if not hasattr(env, '_walkthrough'):
        env._walkthrough = WalkingState(env.num_envs, env.device, env.step_dt)
    s = env._walkthrough
    if s.last_step == int(env.common_step_counter):
        return s
    d = env.scene['robot'].data
    p = d.root_link_pos_w - env.scene.env_origins
    v, q = d.root_link_lin_vel_w, d.root_link_quat_w
    foot = env.scene.sensors['cliff_foot_contact'].data
    forces = [env.scene.sensors[name].data.force for name in
              ('head_ground_contact', 'trunk_ground_contact')]
    finite = torch.isfinite(torch.cat((p, v, q, d.joint_pos, d.joint_vel,
                                      env.action_manager.action), -1)).all(-1)
    for f in forces:
        finite &= torch.isfinite(f).reshape(env.num_envs, -1).all(-1)
    finite &= torch.isfinite(foot.pos).reshape(env.num_envs, -1).all(-1)
    head, trunk = [torch.linalg.vector_norm(f.sum(1), dim=-1) for f in forces]
    s.update(step=int(env.common_step_counter), age=env.episode_length_buf,
             x=p[:, 0], y=p[:, 1], vx=v[:, 0], vz=v[:, 2],
             cos_tilt=1 - 2 * (q[:, 1]**2 + q[:, 2]**2),
             supported=env.scene.sensors['feet_ground_contact'].data.found.reshape(env.num_envs, -1).bool().any(-1),
             lower=lower_floor_contact(foot.found, foot.pos, env.scene.env_origins),
             head=head, trunk=trunk, finite=finite, horizon=env.max_episode_length)
    return s


def reward(env):
    # Weight 1 and division by dt make each pulse pay its stated points.
    s = update(env)
    # Other failure guards (including the inherited NaN guard) also veto a
    # timeout payout. Boundary is a timeout, but is already latched in s.bad.
    valid = ~env.reset_terminated
    return (8 * (s.block_pulse & valid).float()
            + 40 * (s.pulse & valid).float()) / env.step_dt


def boundary(env):
    return update(env).boundary.clone()
