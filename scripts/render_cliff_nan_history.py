"""Replay saved finite poses, without stepping or reconstructing missing controls."""
import argparse
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,default=Path('logs/course-session-20261024'))
    args=parser.parse_args()
    folder=args.session/'artifacts/logs/course-session/nan-dumps'
    output=args.session/'analysis/failure-audit/pre-nan-finite-history.mp4'
    model=mujoco.MjModel.from_binary_path(str(folder/'model_latest.mjb'))
    data=mujoco.MjData(model)
    z=np.load(folder/'nan_dump_latest.npz',allow_pickle=True)
    meta=z['_metadata'].item()
    cam=mujoco.MjvCamera();cam.distance=.8;cam.azimuth=90;cam.elevation=-20
    font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',17)
    # 200 Hz physics samples played at 25 fps = 8x slow motion.
    # The complete clip is 0.5 seconds of recorded history; no NaN pose exists.
    with mujoco.Renderer(model,height=640,width=960) as renderer, imageio.get_writer(output,fps=25,codec='libx264',quality=8) as writer:
        for i,key in enumerate(sorted(k for k in z.files if k.startswith('states_step_'))):
            mujoco.mj_setState(model,data,z[key][0],meta['state_spec'])
            mujoco.mj_fwdPosition(model,data)
            cam.lookat[:]=data.qpos[:3]
            cam.lookat[2] += .04  # Leave space above the complete robot for labels.
            renderer.update_scene(data,camera=cam)
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,960,125),fill='#101820')
            w,x,y,zz=data.qpos[3:7]
            yaw=np.degrees(np.arctan2(2*(w*zz+x*y),1-2*(y*y+zz*zz)))
            lines=[f'PRE-NaN FINITE HISTORY | env 1232 | physics step {key.split("_")[-1]}',
                   f'8x slow motion | {i*.005:.3f} s of 0.500 s | heading {yaw:+.1f} deg',
                   f'WORLD vx={data.qvel[0]:+.3f}  vy={data.qvel[1]:+.3f}  vz={data.qvel[2]:+.3f} m/s',
                   'Saved poses only. Controls and original contact forces NOT recorded.',
                   'Last frame precedes detection; this is not a reproduced NaN event.']
            for j,line in enumerate(lines):draw.text((14,5+j*24),line,font=font,fill='white')
            writer.append_data(np.asarray(frame))
            if i in [0,49,99]:frame.save(output.with_name(f'pre-nan-frame-{i}.png'))
    print(output)


if __name__=='__main__':main()
