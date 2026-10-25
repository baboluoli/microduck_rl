"""Synthetic evidence: failure veto, same-sample contact, pulses and gates."""
import math
import pytest
import torch
from mjlab_microduck.tasks.cliff.course import CourseState, lower_contact, HEIGHTS, STAGES, FINISH
from mjlab_microduck.tasks.cliff.course_curriculum import Curriculum, Result, rank


def tick(s, step=1, **kwargs):
    values = dict(step=step, age=torch.tensor([step]), x=torch.tensor([FINISH]),
                  y=torch.zeros(1), lower=torch.tensor([True]), cos_tilt=torch.ones(1),
                  head=torch.zeros(1), trunk=torch.zeros(1), finite=torch.tensor([True]),
                  failure=torch.tensor([False]))
    values.update(kwargs)
    return s.update(**values)


def test_once_cached_and_staggered_reset():
    s = CourseState(1, 'cpu')
    assert tick(s).item()
    assert tick(s).item()
    assert not tick(s, 2).item()
    s.reset(torch.tensor([0]))
    assert tick(s, 3).item()


@pytest.mark.parametrize('failure', ['head', 'trunk', 'finite', 'failure', 'cos_tilt', 'lower', 'y'])
def test_same_step_veto(failure):
    value = {'head': 16., 'trunk': 21., 'finite': False, 'failure': True,
             'cos_tilt': math.cos(math.pi/4), 'lower': False, 'y': 1.2}[failure]
    assert not tick(CourseState(1, 'cpu'), **{failure:torch.tensor([value])}).item()


@pytest.mark.parametrize('height', HEIGHTS)
def test_contact_geometry(height):
    pos = torch.tensor([[[.6, 0, -height], [.6, 0, 0]]])
    assert lower_contact(torch.tensor([[True, False]]), pos, torch.zeros(1,3), torch.tensor([height])).item()
    assert not lower_contact(torch.tensor([[False, True]]), pos, torch.zeros(1,3), torch.tensor([height])).item()


def test_latched_impact_and_contact_tilt():
    s = CourseState(1,'cpu')
    tick(s, x=torch.tensor([1.]), lower=torch.tensor([False]), head=torch.tensor([16.]))
    assert not tick(s,2).item()
    s = CourseState(1,'cpu')
    tick(s, x=torch.tensor([1.]), cos_tilt=torch.tensor([.7]))
    assert not tick(s,2).item()


@pytest.mark.parametrize('side', [-1., 1.])
def test_boundary_is_strict_and_latched(side):
    # Exactly on the permitted line is valid; crossing it ends the task even
    # if the same sample reaches the finish. Returning inside cannot erase it.
    s = CourseState(1, 'cpu')
    tick(s, x=torch.tensor([1.]), y=torch.tensor([side*1.15]))
    assert not s.boundary.item()
    assert not tick(s, 2, y=torch.tensor([side*1.151])).item()
    assert s.boundary.item()
    assert not tick(s, 3).item()


def test_gates_persistence_regression(tmp_path):
    c = Curriculum()
    batch = {k:Result((22,22,22), 2, 68) for k in c.conditions()}
    assert c.advance(batch, batch, 'source-sha')
    assert c.upper_mm == 22.5
    # Regression now means "fails the gate" on 2 cm or the previous stage.
    bad = {k:Result((18,18,18), 2, 60) for k in c.conditions()}
    assert not c.regression(bad)
    path = tmp_path/'state.json'
    c.save(path)
    c = Curriculum.load(path)
    assert c.regression(bad)
    good = {k:Result((22,22,22), 2, 68) for k in c.conditions()}
    assert not c.regression(good)
    with pytest.raises(ValueError): c.advance({}, {}, 'incomplete')
    assert rank([Result((22,22,22),1,67)]) > rank([Result((22,22,22),2,72)])


def test_config_and_reward_integration():
    from types import SimpleNamespace
    from mjlab_microduck.tasks.cliff.course import reward
    from mjlab_microduck.tasks.cliff.env_cfg import make_microduck_cliff_course_env_cfg
    cfg = make_microduck_cliff_course_env_cfg()
    assert cfg.episode_length_s == 12
    assert cfg.scene.terrain.terrain_generator.num_rows == len(HEIGHTS)
    assert 'cliff_landing' not in cfg.terminations
    assert not cfg.terminations['course_boundary'].time_out
    assert cfg.terminations['time_out'].time_out
    for dt in (.01,.02,.04):
        s = CourseState(1,'cpu')
        tick(s)
        env = SimpleNamespace(_course=s, common_step_counter=1, step_dt=dt)
        assert reward(env).item()*dt == pytest.approx(80)


def test_independent_classifier():
    from mjlab_microduck.tasks.cliff.course_evaluation import classify
    for mode in ('good', 'impact', 'tilt', 'missing', 'boundary'):
        s = CourseState(1,'cpu')
        samples = []
        for i,x in enumerate((.6, 1., 2., FINISH), 1):
            head = 16. if mode == 'impact' and i == 2 else 0.
            cosine = .7 if mode == 'tilt' and i == 1 else 1.
            y = 1.2 if mode == 'boundary' and i == 4 else 0.
            found = mode != 'missing'
            tick(s,i,x=torch.tensor([x]), head=torch.tensor([head]), cos_tilt=torch.tensor([cosine]),
                 y=torch.tensor([y]), lower=torch.tensor([found]))
            samples.append(dict(x=x,y=y,age=i,head=head,trunk=0.,cos_tilt=cosine,finite=True,failure=False,
                feet=[dict(found=found,position=[x,0,-.02],force=[0,0,3]),
                      dict(found=False,position=[0,0,0],force=[0,0,0])]))
        assert classify(samples,height=.02)['success'] == s.pulse.item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires approved GPU with EGL preflight')
@pytest.mark.parametrize('play', [False,True])
def test_cuda_course_reset_geometry(play):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks.cliff.env_cfg import make_microduck_cliff_course_env_cfg
    from mjlab_microduck.tasks.cliff.course import observation
    cfg = make_microduck_cliff_course_env_cfg(play)
    cfg.scene.num_envs = 8
    env = ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        for height in STAGES:
            for distance in (.45,.55,.65):
                env.event_manager.get_term_cfg('course_reset').params.update(fixed_height=height,start_range=(distance,distance))
                env.reset(seed=42)
                expected = torch.tensor([distance,height],device='cuda:0').expand(8,2)
                torch.testing.assert_close(observation(env),expected,atol=2e-5,rtol=0)
                assert env.observation_manager.group_obs_dim['actor'] == (63,)
                assert env.observation_manager.group_obs_dim['critic'] == (78,)
                assert env.action_manager.total_action_dim == 14
                obs, *_ = env.step(torch.zeros(8,14,device='cuda:0'))
                assert all(torch.isfinite(v).all() for v in obs.values())
    finally:
        env.close()


def test_random_reset_moves_full_pose_and_keeps_fixed_edge():
    from types import SimpleNamespace
    from mjlab_microduck.tasks.cliff.course import reset, observation
    n = 10000
    terrain = SimpleNamespace(terrain_levels=torch.zeros(n,dtype=torch.long),terrain_types=torch.zeros(n,dtype=torch.long),
        env_origins=torch.zeros(n,3),terrain_origins=torch.stack([torch.tensor([[i*5.,0.,0.]]) for i in range(21)]))
    class Robot:
        def __init__(self):
            self.data = SimpleNamespace(indexing=SimpleNamespace(free_joint_q_adr=list(range(7))))
        def write_root_link_pose_to_sim(self, pose, env_ids):
            env.sim.data.qpos[env_ids] = pose
            self.data.root_link_pos_w = env.sim.data.qpos[:,:3]
    robot = Robot()
    class Scene:
        def __init__(self): self.terrain=terrain
        @property
        def env_origins(self): return terrain.env_origins
        def __getitem__(self,key): return robot
    env = SimpleNamespace(scene=Scene(),device='cpu',sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(n,7)) ))
    env.sim.data.qpos[:,2] = .2
    env.sim.data.qpos[:,3] = 1
    torch.manual_seed(42)
    reset(env,torch.arange(n),upper=.025)
    obs = observation(env)
    assert .4499 <= obs[:,0].min() and obs[:,0].max() <= .6501
    assert obs[:,0].mean().item() == pytest.approx(.55,abs=.002)
    assert ((obs[:,1] > .0225).float().mean()).item() == pytest.approx(.75,abs=.02)
    torch.testing.assert_close(env.sim.data.qpos[:,2],torch.full((n,),.2))
    torch.testing.assert_close(env.sim.data.qpos[:,3],torch.ones(n))
    assert obs[:,1].min() >= .02 and obs[:,1].max() <= .025


def test_staggered_rows_and_cached_manager_calls():
    s = CourseState(2,'cpu')
    kw = dict(age=torch.tensor([1,5]),x=torch.tensor([FINISH,1.]),y=torch.zeros(2),
              lower=torch.ones(2,dtype=torch.bool),cos_tilt=torch.ones(2),head=torch.zeros(2),
              trunk=torch.zeros(2),finite=torch.ones(2,dtype=torch.bool),failure=torch.zeros(2,dtype=torch.bool))
    assert s.update(step=1,**kw).tolist() == [True,False]
    assert s.update(step=1,**kw).tolist() == [True,False]
    s.reset(torch.tensor([0]))
    kw.update(age=torch.tensor([1,6]),x=torch.tensor([0.,FINISH]),lower=torch.tensor([False,True]))
    assert s.update(step=2,**kw).tolist() == [False,True]
    assert not s.contact[0] and s.contact[1]


def test_precompiled_bank_matches_declared_heights_and_footprint():
    import mujoco
    import numpy as np
    from mjlab_microduck.tasks.cliff.terrain import CliffDropTerrainCfg
    for row,height in enumerate(HEIGHTS):
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name='terrain')
        tile = CliffDropTerrainCfg(size=(5.,3.),height_buckets=HEIGHTS,min_drop=HEIGHTS[0],max_drop=HEIGHTS[-1])
        tile.function((row+.5)/len(HEIGHTS),spec,np.random.default_rng(42))
        model = spec.compile()
        # Upper edge at 1.5 absolute; spawn .95 means signed edge .55.
        assert model.geom_pos[0,0]+model.geom_size[0,0]-.95 == pytest.approx(.55)
        assert model.geom_pos[1,0]+model.geom_size[1,0]-.95 == pytest.approx(4.05)
        assert model.geom_pos[1,2]+model.geom_size[1,2] == pytest.approx(-height)
        assert 4.05-FINISH == pytest.approx(.5)


def test_full_course_bank_has_no_border_overlap():
    """Raycast the complete bank, including its border, on both sides of spawn.

    Testing a single isolated box missed a real bug: the tile was centered on
    the grid corner, and the outer flat border covered half its lower floor.
    Rays measure the surface physics will actually collide with.
    """
    import mujoco
    import numpy as np
    from mjlab.terrains.terrain_generator import TerrainGenerator
    from mjlab_microduck.tasks.cliff.env_cfg import make_microduck_cliff_course_env_cfg
    cfg = make_microduck_cliff_course_env_cfg(play=True)
    generator = TerrainGenerator(cfg.scene.terrain.terrain_generator, device='cpu')
    spec = mujoco.MjSpec()
    generator.compile(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for row, height in enumerate(HEIGHTS):
        origin = generator.terrain_origins[row, 0]
        for x in (.75, 2., 3.55):
            for y in (-1.1, -.2, .2, 1.1):
                point = origin + np.array([x, y, 1.])
                distance = mujoco.mj_ray(model, data, point, np.array([0., 0., -1.]),
                                         None, 1, -1, np.array([-1], dtype=np.int32))
                assert point[2] - distance == pytest.approx(-height, abs=1e-7)


def _drift_env(vx, vy, yaw=0., n=1):
    from types import SimpleNamespace
    half = torch.tensor(yaw / 2)
    quat = torch.stack([torch.cos(half), torch.zeros(()), torch.zeros(()), torch.sin(half)]).repeat(n, 1)
    data = SimpleNamespace(root_link_quat_w=quat, root_link_lin_vel_w=torch.tensor([[vx, vy, 0.]] * n))
    return SimpleNamespace(scene={'robot': SimpleNamespace(data=data)}, num_envs=n, device='cpu',
                           step_dt=.02, episode_length_buf=torch.ones(n, dtype=torch.long))


def test_sideways_drift_charges_steady_crab_not_sway():
    from mjlab_microduck.tasks.cliff.course import sideways_drift
    from mjlab_microduck.tasks.cliff.env_cfg import make_microduck_cliff_course_env_cfg
    term = make_microduck_cliff_course_env_cfg().rewards['sideways_drift']
    assert term.weight > 0  # Self-negating penalty: positive weight, never a reward.
    crab, sway = _drift_env(.2, -.13), _drift_env(.2, 0.)
    for step in range(300):  # 6 s at 50 Hz.
        if step:
            crab.episode_length_buf += 1; sway.episode_length_buf += 1
        sway.scene['robot'].data.root_link_lin_vel_w[:, 1] = .15 * math.sin(2 * math.pi * 2 * step * .02)  # 2 Hz gait sway.
        c, w = sideways_drift(crab), sideways_drift(sway)
    assert c.item() == pytest.approx(-.13, abs=.005)  # Steady drift is charged in full.
    assert abs(w.item()) < .02  # Oscillation largely cancels.
    assert (c <= 0).all() and (w <= 0).all()


def test_sideways_drift_uses_heading_frame_and_resets():
    from mjlab_microduck.tasks.cliff.course import sideways_drift
    yaw = math.radians(30)  # Turned body walking the way it faces: world velocity along the heading.
    env = _drift_env(.2 * math.cos(yaw), .2 * math.sin(yaw), yaw=yaw)
    for step in range(200):
        if step:
            env.episode_length_buf += 1
        value = sideways_drift(env)
    assert value.item() == pytest.approx(0, abs=1e-6)  # Facing direction is the heading controller's job.
    env.scene['robot'].data.root_link_lin_vel_w[:] = torch.tensor([[0., .2, 0.]])
    for _ in range(100):
        env.episode_length_buf += 1; sideways_drift(env)
    assert sideways_drift(env).item() < -.05
    env.episode_length_buf[:] = 1  # New episode: no penalty carried over.
    env.scene['robot'].data.root_link_lin_vel_w[:] = torch.tensor([[.2 * math.cos(yaw), .2 * math.sin(yaw), 0.]])
    assert sideways_drift(env).item() == pytest.approx(0, abs=1e-6)
