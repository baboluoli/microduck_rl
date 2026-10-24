"""Offline audit of the immutable 2026-10-24 course artifacts; never simulates/trains.

Velocity derived from positions is an interval average in WORLD coordinates.
It cannot distinguish sideways walking from forward walking at a wrong heading.
The MP4s deliberately use a dot for the robot: joint poses were not recorded.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

DT = .02


def series(row):
    t = row['trace']
    xy = np.array([[s['x'], s['y']] for s in t])
    time = np.array([s['age'] * DT for s in t])
    # Keep the first velocity unknown; copying the next interval fabricates data.
    vy = np.r_[np.nan, np.diff(xy[:, 1]) / np.diff(time)]
    return time, xy, vy


def episode_metrics(row):
    time, xy, vy = series(row)
    edge = np.flatnonzero(xy[:, 0] >= .55)
    e = int(edge[0]) if len(edge) else len(xy)-1
    phases = {'approach': xy[:, 0] < .55, 'early_lower': (xy[:, 0] >= .55) & (xy[:, 0] < 1.55), 'late_lower': xy[:, 0] >= 1.55}
    out = dict(env=row['env'], seed=row['seed'], height=row['height'], distance=row['distance'], outcome=row['outcome'], success=row['success'], end_x=xy[-1,0], end_y=xy[-1,1], edge_time=time[e], edge_y=xy[e,1], approach_dy=xy[e,1]-xy[0,1], duration=time[-1])
    for name, mask in phases.items():
        out[name+'_vy'] = float(np.nanmean(vy[mask])) if np.isfinite(vy[mask]).any() else None
    # A sustained negative drift means net >1 cm displacement over 0.5 s.
    # This is a diagnostic convention, not a trained failure threshold.
    window = 25
    hits = np.flatnonzero((xy[window:,1]-xy[:-window,1]) / (window*DT) < -.02)
    out['negative_drift_window_start_s'] = float(time[hits[0]]) if len(hits) else None
    return out


def trace_video(source, trained, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), dpi=100)
    ax, av = axes
    ax.set(xlim=(-.25,3.9), ylim=(-1.4,1.4), xlabel='Forward position X (m)', ylabel='World Y (m)', title='Recorded first-episode trajectories (dot = root position)')
    ax.axvline(.55, color='gray', ls='--', label='Drop edge')
    ax.axvline(3.55, color='green', ls='--', label='Finish')
    for y in [-1.15,1.15]: ax.axhline(y,color='red',ls=':')
    all_vy = np.concatenate([series(r)[2] for r in (source, trained)])
    av.set(xlim=(0,max(source['duration'],trained['duration'])), ylim=(min(-.1,float(np.nanmin(all_vy)))-.05,max(.1,float(np.nanmax(all_vy)))+.05), xlabel='Time (s)', ylabel='World sideways speed (m/s)')
    av.axhline(0,color='gray',lw=1)
    curves=[]
    for row,color,label in [(source,'#2383c4','Source 249'),(trained,'#e46d26','After 100 updates')]:
        time, xy, vy=series(row)
        line,=ax.plot([],[],color=color,label=label)
        dot,=ax.plot([],[],'o',color=color)
        vl,=av.plot([],[],color=color,label=label)
        curves.append((row,time,xy,vy,line,dot,vl))
    ax.legend(loc='upper right');av.legend(loc='upper right')
    title=fig.suptitle('')
    fig.text(.07,.015,'Velocity = change in saved Y / 0.02 s; not body-frame speed. No recorded heading or joint poses.',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.94))
    with imageio.get_writer(path,fps=25,codec='libx264',quality=7,macro_block_size=2) as writer:
        for now in np.arange(.02,max(source['duration'],trained['duration'])+.041,.04):
            details=[]
            for row,time,xy,vy,line,dot,vl in curves:
                k=max(1,int(np.searchsorted(time,now,side='right')))
                line.set_data(xy[:k,0],xy[:k,1]);dot.set_data([xy[k-1,0]],[xy[k-1,1]])
                vl.set_data(time[:k],vy[:k])
                details.append(f"{'SOURCE' if row is source else 'TRAINED'} vy={vy[k-1]:+.3f} m/s" + (f" [{row['outcome']}; ENDED]" if now>=time[-1] else ''))
            title.set_text(f"h={trained['height']*100:.2f} cm | start={trained['distance']:.2f} m | seed={trained['seed']} env={trained['env']} | t={now:.2f}s\n"+'    '.join(details))
            fig.canvas.draw();writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3])
    fig.savefig(path.with_suffix('.png'));plt.close(fig)


def nan_audit(root,out):
    import mujoco
    folder=root/'course-session/nan-dumps'
    # This is our verified local artifact; metadata is a pickled scalar dict.
    z=np.load(folder/'nan_dump_latest.npz',allow_pickle=True)
    meta=z['_metadata'].item()
    model=mujoco.MjModel.from_binary_path(str(folder/'model_latest.mjb'))
    data=mujoco.MjData(model)
    records=[]
    for key in sorted(k for k in z.files if k.startswith('states_step_')):
        mujoco.mj_setState(model,data,z[key][0],meta['state_spec'])
        # Kinematics/collision only. Do not advance physics or invent controls.
        mujoco.mj_fwdPosition(model,data)
        w,x,y,zz=data.qpos[3:7]
        heading=np.arctan2(2*(w*zz+x*y),1-2*(y*y+zz*zz))
        contacts=[(float(c.dist),int(c.geom1),int(c.geom2)) for c in data.contact]
        deepest=min(contacts,default=(0,-1,-1))
        records.append(dict(step=int(key.split('_')[-1]),heading=float(heading),root_speed=float(np.linalg.norm(data.qvel[:3])),angular_speed=float(np.linalg.norm(data.qvel[3:6])),joint_speed=float(np.max(np.abs(data.qvel[6:]))),vx=float(data.qvel[0]),vy=float(data.qvel[1]),z=float(data.qpos[2]),contacts=len(contacts),min_distance=deepest[0],deepest_pair=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) if i>=0 else None for i in deepest[1:]]))
    result=dict(metadata=meta,physics_dt=model.opt.timestep,seconds=len(records)*model.opt.timestep,controls_saved=bool(meta['state_spec']&int(mujoco.mjtState.mjSTATE_CTRL)),original_contact_forces_saved=False,quarters=[{k: max(r[k] for r in chunk) for k in ['root_speed','angular_speed','joint_speed','contacts']} for chunk in np.array_split(np.array(records,dtype=object),4)],deepest_contact=min(records,key=lambda r:r['min_distance']),records=records)
    (out/'nan-forensics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def normalizers(root,out):
    import torch
    from rsl_rl.modules.normalization import EmpiricalNormalization
    result={}
    files=[root/'rsl_rl/oracle_course/course-session-source/model_source.pt',root/'rsl_rl/oracle_course/course-session-source/model_staged.pt',*sorted(root.glob('rsl_rl/oracle_course/*course-100/model_*.pt'))]
    eps=EmpiricalNormalization(63).eps
    for f in files:
        checkpoint=torch.load(f,map_location='cpu',weights_only=False)
        result[str(f.relative_to(root))]={}
        for group in ['actor','critic']:
            d=checkpoint[group+'_state_dict'];mean=float(d['obs_normalizer._mean'][0,-1]);std=float(d['obs_normalizer._std'][0,-1])
            result[str(f.relative_to(root))][group]=dict(mean=mean,std=std,eps=eps,count=int(d['obs_normalizer.count']),normalized_heights={str(h):(h-mean)/(std+eps) for h in [.02,.0205,.021,.0215,.022,.0225]})
    (out/'height-normalization.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,default=Path('logs/course-session-20261024'))
    parser.add_argument('--videos',action='store_true')
    args=parser.parse_args();root=args.session/'artifacts/logs';out=args.session/'analysis/failure-audit';out.mkdir(parents=True,exist_ok=True)
    sets={};metrics=[]
    for label,folder in [('source','eval-005'),('trained','eval-006')]:
        rows=[]
        for file in sorted((root/'course-session'/folder).glob('h*.json')):
            rows.extend(json.loads(file.read_text()))
        sets[label]={(r['height'],r['distance'],r['seed'],r['env']):r for r in rows}
        metrics.extend(dict(policy=label,**episode_metrics(r)) for r in rows)
    with (out/'episodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    summary={}
    for label in sets:
        group=[m for m in metrics if m['policy']==label]
        summary[label]=dict(n=len(group),success=sum(m['success'] for m in group),negative_boundary=sum(int(m['outcome']=='boundary' and m['end_y']<0) for m in group),positive_boundary=sum(int(m['outcome']=='boundary' and m['end_y']>0) for m in group),means={k:float(np.mean([m[k] for m in group if m[k] is not None])) for k in ['edge_y','approach_dy','approach_vy','early_lower_vy','late_lower_vy','negative_drift_window_start_s']})
    # Select median-duration negative-Y boundary failure at each start distance,
    # 2 cm only, among episodes whose source counterpart succeeded. Failure
    # selection is deliberate and disclosed, never used as a success-rate estimate.
    selected=[]
    for distance in [.45,.55,.65]:
        eligible=[(k,r) for k,r in sets['trained'].items() if k[0]==.02 and k[1]==distance and r['outcome']=='boundary' and r['trace'][-1]['y']<0 and sets['source'][k]['success']]
        eligible.sort(key=lambda pair:pair[1]['duration'])
        key,row=eligible[len(eligible)//2];filename=f'failure-d{distance:.2f}-s{row["seed"]}-env{row["env"]}.mp4'
        selected.append(dict(key=key,file=filename,source=episode_metrics(sets['source'][key]),trained=episode_metrics(row)))
        if args.videos: trace_video(sets['source'][key],row,out/filename)
    summary['video_selection']='Median-duration negative-Y boundary failure per start distance at 2 cm with a successful matched source; same seed/env does not guarantee identical stochastic histories.'
    summary['videos']=selected
    summary['missing_telemetry']=['heading','requested angular velocity','actual angular velocity','body-frame sideways velocity']
    summary['early_mean_y'] = {label: {str(t): float(np.mean([r['trace'][round(t/DT)-1]['y'] for r in rows.values()])) for t in [.1,.2,.5,1.]} for label,rows in sets.items()}
    # Every reported boundary failure must first breach a limit on its final
    # pre-reset sample, and must not receive the completion pulse.
    for rows in sets.values():
        for row in rows.values():
            breaches=[i for i,s in enumerate(row['trace']) if s['x'] < -.75 or s['x'] > 3.8 or abs(s['y']) > 1.15]
            if row['outcome']=='boundary':
                assert breaches == [len(row['trace'])-1]
                assert not any(s['pulse'] for s in row['trace'])
    summary['boundary_trace_assertions_passed'] = True
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    nan_audit(root,out);normalizers(root,out)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
