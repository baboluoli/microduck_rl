"""Course completion, with tile-relative geometry and control-step cached state.

Metres, seconds and newtons throughout. This experimental task retains the
63-input oracle interface; it is not a hardware deployment policy.
"""
import math
import torch

# Precompiled terrain rows, one per drop height. Built from integer tenths of
# a millimetre so that literals such as .0225 or .035 compare exactly. The
# original 2–3 cm rows at 0.5 mm come first and never move: old checkpoints'
# row indices and saved ``course_height_counts`` keep their meaning. Taller
# drops, 32.5–100 mm, are appended at 2.5 mm (49 rows), then 105–140 mm at
# 5 mm (57 rows) and 150–240 mm at 10 mm (67 rows in total). Every row is a real terrain tile in every simulated
# world, so the bank is kept small: each stage's band still holds 2–4 practice
# heights. Rows are only ever appended, so older checkpoints keep their meaning.
HEIGHTS = tuple(t / 10000 for t in (*range(200, 301, 5), *range(325, 1001, 25), *range(1050, 1401, 50),
                                    *range(1500, 2401, 100)))
# Curriculum ladder. Each step is roughly 15–25% harder than the one before;
# fixed 0.25 cm steps would be tiny near 10 cm and need 32 stages. Above 10 cm
# the steps are larger (14, 18, 24 cm: 30–40% harder each), to climb faster.
# The 10 cm policy already lands 17 of 24 untrained 24 cm drops (one CPU seed,
# 2026-10-26), so the jumps are not blind. 24 cm is about the robot's height.
STAGES = (.02, .0225, .025, .03, .035, .04, .05, .06, .07, .08, .09, .10, .14, .18, .24)
EDGE = .55
# Historical finish line (3 m past the edge). No longer the success rule; the
# lower floor still extends past it, up to FORWARD_EDGE.
FINISH = 3.55
# Success: after the first lower-floor touchdown, walk this far forward
# without an impact, tilt, physical failure or leaving the course. There is
# no finish line to reach; this only rules out "land, then stand still".
GOAL = 1.
FORWARD_EDGE = 4.05
SIDE_LIMIT = 1.15


def lower_contact(found, positions, origins, heights):
    p = positions.reshape(len(origins), -1, 3) - origins[:, None, :]
    return (found.reshape(len(origins), -1).bool() & torch.isfinite(p).all(-1)
            & (p[..., 0] >= EDGE) & (p[..., 0] <= FORWARD_EDGE)
            & (p[..., 1].abs() <= 1.35)
            & ((p[..., 2] + heights[:, None]).abs() <= .008)).any(-1)


class CourseState:
    def __init__(self, n, device):
        self.last_step = -1
        for name in ('contact', 'impact', 'tilt', 'nonfinite', 'boundary', 'bad', 'paid', 'pulse'):
            setattr(self, name, torch.zeros(n, dtype=torch.bool, device=device))
        self.contact_x = torch.full((n,), float('nan'), device=device)
        # Pay-as-you-go completion credit: the furthest X already paid for,
        # and the distance paid on the latest control step (metres).
        self.paid_x = torch.full((n,), float('nan'), device=device)
        self.gain = torch.zeros(n, device=device)

    def reset(self, ids):
        for name in ('contact', 'impact', 'tilt', 'nonfinite', 'boundary', 'bad', 'paid', 'pulse'):
            getattr(self, name)[ids] = False
        self.contact_x[ids] = float('nan')
        self.paid_x[ids] = float('nan')
        self.gain[ids] = 0

    def update(self, *, step, age, x, y, lower, cos_tilt, head, trunk, finite, failure, cap=None):
        """Advance the latched course state by one control step.

        ``cap`` limits the distance paid per step (commanded speed x control
        dt). None pays any forward distance; tests use that default.
        """
        if self.last_step == step:
            return self.pulse
        self.last_step = step
        self.reset(age <= 1)
        first = lower & ~self.contact
        self.contact_x[first] = x[first]
        self.contact |= lower
        self.impact |= (head > 15) | (trunk > 20)
        self.nonfinite |= ~finite
        # Contact is latched first: tilt on the first touchdown sample counts.
        self.tilt |= self.contact & (cos_tilt <= math.cos(math.pi / 4))
        self.boundary |= (x < -.75) | (x > 3.8) | (y.abs() > SIDE_LIMIT)
        self.bad |= self.impact | self.nonfinite | self.tilt | self.boundary | failure
        self.pulse = self.contact & (x - self.contact_x >= GOAL) & ~self.bad & ~self.paid
        self.paid |= self.pulse
        # Distance credit starts where the foot first touched the lower floor
        # and stops GOAL metres later. Only new ground counts (a high-water mark),
        # so stepping back and forth earns nothing. Each step pays at most
        # ``cap``; faster travel becomes a backlog that is paid only if the
        # robot later slows down. Success ends the episode, so a robot that
        # rushes loses its backlog: walking at the commanded speed pays best.
        self.paid_x = torch.where(first, x, self.paid_x)
        paying = self.contact & ~self.bad
        ahead = torch.nan_to_num(torch.minimum(x, self.contact_x + GOAL) - self.paid_x).clamp_min(0)
        if cap is not None:
            ahead = torch.minimum(ahead, cap)
        self.gain = torch.where(paying, ahead, torch.zeros_like(ahead))
        self.paid_x = torch.where(paying, self.paid_x + self.gain, self.paid_x)
        return self.pulse


def heights(env):
    bank = torch.tensor(HEIGHTS, device=env.device)
    return bank[env.scene.terrain.terrain_levels]


def observation(env):
    x = env.scene['robot'].data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return torch.stack((EDGE - x, heights(env)), -1)


def bands(upper):
    """Row index ranges ``(easier_end, frontier_end)`` for a curriculum stage.

    Frontier rows are the heights newly unlocked by this stage
    (previous stage < h <= upper), i.e. rows ``easier_end .. frontier_end-1``.
    Easier rows are everything up to and including the previous stage,
    rows ``0 .. easier_end-1``. At the first stage (2 cm) there is no frontier.
    """
    if upper not in STAGES:
        raise ValueError('Unsupported course height')
    stage = STAGES.index(upper)
    if stage == 0:
        return 1, 1
    previous = STAGES[stage - 1]
    return (sum(h <= previous for h in HEIGHTS), sum(h <= upper for h in HEIGHTS))


def reset(env, env_ids, upper=.02, start_range=(.45, .65), fixed_height=None):
    """Move the complete reset robot between precompiled tiles, after base reset.

    Geometry stays immutable. Only these episode rows change tile or approach.
    The ordinary reset has already initialized joint state and root velocity.
    """
    if fixed_height is not None and fixed_height not in HEIGHTS:
        raise ValueError('Unsupported course height')
    easier_end, frontier_end = bands(upper)
    terrain = env.scene.terrain
    old = env.scene.env_origins[env_ids].clone()
    n = len(env_ids)
    if fixed_height is not None:
        levels = torch.full((n,), HEIGHTS.index(fixed_height), device=env.device, dtype=torch.long)
    else:
        # 75% of resets practise the newly unlocked heights, 25% revisit the
        # easier ones so they are not forgotten. Rows are sampled uniformly
        # inside each band. The call order matches the original 2–3 cm sampler,
        # so stage .0225 draws exactly the same rows for the same RNG state.
        frontier = torch.randint(easier_end, max(easier_end + 1, frontier_end), (n,), device=env.device)
        easier = torch.randint(0, easier_end, (n,), device=env.device)
        levels = (torch.where(torch.rand(n, device=env.device) < .75, frontier, easier)
                  if frontier_end > easier_end else torch.zeros_like(easier))
    # Keep a cumulative histogram of actual reset samples. The runner saves
    # these counts with each checkpoint, so the sampler can be audited later.
    if not hasattr(env, '_course_height_counts'):
        env._course_height_counts = torch.zeros(len(HEIGHTS), dtype=torch.long, device=env.device)
    env._course_height_counts += torch.bincount(levels, minlength=len(HEIGHTS))
    terrain.terrain_levels[env_ids] = levels
    terrain.env_origins[env_ids] = terrain.terrain_origins[levels, terrain.terrain_types[env_ids]]
    robot = env.scene['robot']
    adr = robot.data.indexing.free_joint_q_adr
    pose = env.sim.data.qpos[env_ids][:, adr].clone()
    pose[:, :3] += env.scene.env_origins[env_ids] - old
    distance = torch.empty(n, device=env.device).uniform_(*start_range)
    pose[:, 0] += EDGE - distance
    robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    if hasattr(env, '_course'):
        env._course.reset(env_ids)


def update(env):
    if not hasattr(env, '_course'):
        env._course = CourseState(env.num_envs, env.device)
    s = env._course
    if s.last_step == int(env.common_step_counter):
        return s
    d = env.scene['robot'].data
    p = d.root_link_pos_w - env.scene.env_origins
    q = d.root_link_quat_w
    foot = env.scene.sensors['cliff_foot_contact'].data
    forces = [env.scene.sensors[name].data.force for name in ('head_ground_contact', 'trunk_ground_contact')]
    values = [p, q, d.root_link_lin_vel_w, d.root_link_ang_vel_w, d.joint_pos, d.joint_vel, env.action_manager.action,
              foot.pos.reshape(env.num_envs, -1), foot.force.reshape(env.num_envs, -1), *[f.reshape(env.num_envs, -1) for f in forces]]
    finite = torch.isfinite(torch.cat(values, -1)).all(-1)
    head, trunk = [torch.linalg.vector_norm(f.sum(1), dim=-1) for f in forces]
    # Physical guards precede course_success in the configured manager. Read
    # their current results rather than calling stateful fall timers twice.
    failure = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in env.termination_manager.active_terms:
        if name == 'course_success':
            break
        cfg = env.termination_manager.get_term_cfg(name)
        if not cfg.time_out:
            failure |= env.termination_manager.get_term(name)
    # Commanded forward speed; the floor avoids a zero cap for a stop command.
    cap = env.command_manager.get_command('twist')[:, 0].clamp_min(.01) * env.step_dt
    s.update(step=int(env.common_step_counter), age=env.episode_length_buf,
             x=p[:, 0], y=p[:, 1], lower=lower_contact(foot.found, foot.pos, env.scene.env_origins, heights(env)),
             cos_tilt=1-2*(q[:, 1]**2+q[:, 2]**2), head=head, trunk=trunk, finite=finite, failure=failure, cap=cap)
    return s


def success(env):
    return update(env).pulse.clone()


def boundary(env):
    return update(env).boundary.clone()


def sideways_drift(env, tau_s=1.):
    """Penalize slow sideways walking ("crab drift"), not the gait's natural sway.

    The course rewards only world-X progress, so sideways travel used to be
    free until the robot left the course 1.15 m off-centre. The 2026-10-24
    telemetry showed the drift is stepping sideways while facing forward
    (source -0.056 m/s, model_99 -0.132 m/s), not turning.

    Velocity is taken in the heading frame (yaw only), so tilting during the
    drop does not leak vertical speed into "sideways". A biped sways left and
    right every step; a ~1 s moving average (EMA) cancels that oscillation and
    leaves the steady drift, which is the only part the policy should pay for.
    Self-negating (returns <= 0, in m/s): use a POSITIVE weight.
    """
    d = env.scene['robot'].data
    q = d.root_link_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2]**2 + q[:, 3]**2))
    v = d.root_link_lin_vel_w
    lateral = torch.nan_to_num(-torch.sin(yaw) * v[:, 0] + torch.cos(yaw) * v[:, 1])
    if not hasattr(env, '_course_lateral_ema'):
        env._course_lateral_ema = torch.zeros(env.num_envs, device=env.device)
    ema = env._course_lateral_ema
    ema[env.episode_length_buf <= 1] = 0  # First control step of a new episode.
    ema += env.step_dt / tau_s * (lateral - ema)
    return -ema.abs()


# Total completion budget (points), paid per metre over the GOAL after landing.
# Until 2026-10-25 the whole 80 was one lump at a finish line 3 m past the
# edge, which the commanded pace could not reach within 12 s. A reward is
# worth more the sooner it arrives, so the lump paid for rushing: the robot
# sped up toward the finish (0.47 -> 0.80 m/s against a 0.23 m/s command) and
# its walking degraded over consecutive training blocks.
COMPLETION_POINTS = 80
POINTS_PER_METRE = COMPLETION_POINTS / GOAL


def reward(env):
    # The reward manager multiplies by dt, so dividing here pays exactly
    # POINTS_PER_METRE per metre of credited distance.
    return POINTS_PER_METRE * update(env).gain / env.step_dt
