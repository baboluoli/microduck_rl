"""Split a course evaluation's sideways drift into "turned" vs "walked sideways".

Usage (from the RL directory, on evaluation folders written by eval_cliff_course.py):
    uv run python scripts/analyze_course_drift.py logs/.../eval-source logs/.../eval-new-model99

World sideways velocity decomposes as vy_world = vx_body*sin(heading) + vy_body*cos(heading).
The first term is drift from facing the wrong way; the second is crab-walking.
Only the approach before the drop (course X < 0.55 m) is used, skipping the
first 5 control steps after reset. Negative Y means drifting to the robot's right.
"""
import argparse
import glob
import json
import math

import numpy as np


def drift(directory):
    rows = [e for f in sorted(glob.glob(f'{directory}/h*.json')) for e in json.load(open(f))]
    out = dict(heading=[], heading_at_edge=[], vy_world=[], from_turning=[], from_crabbing=[], yaw_cmd=[], yaw_actual=[])
    for e in rows:
        t = [s for s in e['trace'] if s.get('finite', True) and s['x'] < .55 and s['age'] > 5]
        if not t:
            continue
        h = np.array([s['heading'] for s in t])
        vb = np.array([s['velocity_body'] for s in t])
        out['heading'].append(math.degrees(h.mean()))
        out['heading_at_edge'].append(math.degrees(t[-1]['heading']))
        out['vy_world'].append(np.mean([s['vy'] for s in t]))
        out['from_turning'].append(np.mean(vb[:, 0] * np.sin(h)))
        out['from_crabbing'].append(np.mean(vb[:, 1] * np.cos(h)))
        out['yaw_cmd'].append(np.mean([s['command'][2] for s in t]))
        out['yaw_actual'].append(np.mean([s['angular_velocity_world'][2] for s in t]))
    summary = {k: float(np.mean(v)) for k, v in out.items()}
    summary['episodes'] = len(out['heading'])
    summary['successes'] = sum(bool(e['success']) for e in rows)
    summary['boundary_exits'] = sum(e.get('outcome') == 'boundary' for e in rows)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('directories', nargs='+')
    for d in p.parse_args().directories:
        s = drift(d)
        print(f"{d.rstrip('/').split('/')[-1]}: {s['successes']}/{s['episodes']} successes, {s['boundary_exits']} boundary exits; "
              f"heading {s['heading']:+.2f} deg; vy_world {s['vy_world']:+.4f} m/s = turning {s['from_turning']:+.4f} "
              f"+ crabbing {s['from_crabbing']:+.4f}; yaw cmd {s['yaw_cmd']:+.4f} vs actual {s['yaw_actual']:+.4f} rad/s")
