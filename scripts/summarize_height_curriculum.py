"""One row per height-curriculum check, read from a session output folder.

Usage (from the RL directory):
    uv run python scripts/summarize_height_curriculum.py logs/height-curriculum-YYYYMMDD/artifacts/logs/height-curriculum

Columns: block, top height, frontier successes per start (development seeds
20–22, out of 72), impacts / falls / side exits / timeouts summed over the
frontier, regression results (successes out of 72, P = passes the gate) and
whether the policy advanced (or, in hold-height mode, would have). Crabbing
per stage comes from ``analyze_course_drift.py`` on the frontier evaluation
folders.

A second table shows walking quality per block (block 0 is the source
baseline): action std, forward speed before the edge and after landing, wobble
and time to goal. These are recorded in ``stages.jsonl`` by the runner; see
``walking_quality`` for the definitions.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from analyze_course_drift import drift


def action_std(log_text):
    """The last ``Mean action std:`` value in a training log (exploration noise)."""
    values = re.findall(r'Mean action std:\s*([0-9.]+)', log_text)
    return float(values[-1]) if values else None


def checkpoint_action_std(checkpoint):
    """Mean action std stored in a checkpoint, for the source (no training log).

    rsl_rl keeps it as ``distribution.std_param``; the training log prints its mean.
    """
    import torch
    actor = torch.load(checkpoint, map_location='cpu', weights_only=False)['actor_state_dict']
    return float(actor['distribution.std_param'].mean())


def walking_quality(directory):
    """Speed, wobble and time to goal from an evaluation folder's traces.

    - ``speed_before_edge``: body-forward speed (``velocity_body[0]``) on the
      approach, x < 0.5 m and age > 10 steps (skips the first 0.2 s after spawn);
    - ``speed_after_landing``: the same for x > 0.8 m, past the drop;
    - ``yaw_rate_before_edge``: mean |yaw rate| (``angular_velocity_body[2]``)
      on the approach: how much the trunk wobbles side to side;
    - ``time_to_goal_s``: mean ``duration`` of successful episodes.

    Each is averaged per episode first, then across episodes. The command is
    0.18–0.24 m/s, so a speed well above 0.24 means the robot is rushing.
    """
    rows = [r for p in sorted(Path(directory).glob('h*-d*-s*.json')) for r in json.loads(p.read_text())]
    before, after, yaw = [], [], []
    for r in rows:
        trace = [s for s in r['trace'] if s.get('finite', True)]
        approach = [s for s in trace if s['x'] < .5 and s['age'] > 10]
        landed = [s for s in trace if s['x'] > .8]
        if approach:
            before.append(np.mean([s['velocity_body'][0] for s in approach]))
            yaw.append(np.mean([abs(s['angular_velocity_body'][2]) for s in approach]))
        if landed:
            after.append(np.mean([s['velocity_body'][0] for s in landed]))
    durations = [r['duration'] for r in rows if r['success']]
    mean = lambda v: float(np.mean(v)) if v else None  # noqa: E731
    return dict(speed_before_edge=mean(before), speed_after_landing=mean(after),
                yaw_rate_before_edge=mean(yaw), time_to_goal_s=mean(durations))


def outcomes(directory):
    """Count episode outcomes (success / failure / boundary / timeout) and impacts."""
    rows = [r for p in sorted(Path(directory).glob('h*-d*-s*.json')) for r in json.loads(p.read_text())]
    out = dict(episodes=len(rows), impacts=sum('impact' in r['events'] for r in rows),
               falls=sum('tilt' in r['events'] or 'physical_failure' in r['events'] for r in rows))
    for name in ('success', 'failure', 'boundary', 'timeout'):
        out[name] = sum(r['outcome'] == name for r in rows)
    return out


def summarize(session):
    session = Path(session)
    entries = [json.loads(line) for line in (session / 'stages.jsonl').read_text().splitlines() if line.strip()]
    folders = sorted(session.glob('eval-*'))
    # The runner writes one frontier and one regression folder per check, plus
    # a confirmation folder when the frontier passed, in block order.
    frontier = [f for f in folders if f.name.endswith('-frontier')]
    confirmation = iter(f for f in folders if f.name.endswith('-confirmation'))
    rows = []
    for entry, directory in zip(entries, frontier):
        dev = entry['development']
        upper = f"{entry['upper_mm']:g}"
        counts = outcomes(directory)
        try:
            crab = drift(directory)['from_crabbing']
        except (KeyError, ZeroDivisionError, ValueError):
            crab = None
        row = dict(block=entry['block'], upper_mm=entry['upper_mm'],
                   frontier={k.split('/')[1]: sum(v['successes']) for k, v in dev.items() if k.split('/')[0] == upper},
                   impacts=counts['impacts'], falls=counts['falls'], side_exits=counts['boundary'],
                   timeouts=counts['timeout'], crabbing_m_s=crab,
                   regression={k: (sum(dev[k]['successes']), entry['development_passes'][k])
                               for k in entry['regression_streaks']},
                   confirmed=None if entry['confirmation'] is None else all(entry['confirmation_passes'].values()),
                   advanced=entry['advanced'], would_advance=entry.get('would_advance'),
                   quality=entry.get('quality') or {}, frontier_folder=directory.name)
        if entry['confirmation'] is not None:
            row['confirmation_folder'] = next(confirmation).name
        rows.append(row)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('session', type=Path)
    p.add_argument('--json', action='store_true', help='Print machine-readable rows instead of a table')
    args = p.parse_args()
    rows = summarize(args.session)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print('block  top mm  frontier 45/55/65 (/72)  impacts falls side timeouts  crab m/s  regression            confirmed advanced')
        for r in rows:
            f = '/'.join(str(r['frontier'].get(d, '-')) for d in ('45', '55', '65'))
            reg = ' '.join(f"{k}:{n}{'P' if ok else 'F'}" for k, (n, ok) in r['regression'].items())
            crab = '   -   ' if r['crabbing_m_s'] is None else f"{r['crabbing_m_s']:+.3f}"
            print(f"{r['block']:>5}  {r['upper_mm']:>6g}  {f:>23}  {r['impacts']:>7} {r['falls']:>5} {r['side_exits']:>4} "
                  f"{r['timeouts']:>8}  {crab:>8}  {reg:<20}  {str(r['confirmed']):>9} {str(r['advanced']):>8}"
                  + ('' if r['would_advance'] is None else f"  would_advance={r['would_advance']}"))
        print()
        print('block  action std  speed before edge  speed after landing  |yaw rate|  time to goal   (m/s, rad/s, s)')
        number = lambda v, f: '-' if v is None else format(v, f)  # noqa: E731
        for r in rows:
            q = r['quality']
            print(f"{r['block']:>5}  {number(q.get('action_std'), '.2f'):>10}  {number(q.get('speed_before_edge'), '.3f'):>17}"
                  f"  {number(q.get('speed_after_landing'), '.3f'):>19}  {number(q.get('yaw_rate_before_edge'), '.2f'):>10}"
                  f"  {number(q.get('time_to_goal_s'), '.2f'):>12}")
