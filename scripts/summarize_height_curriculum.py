"""One row per height-curriculum check, read from a session output folder.

Usage (from the RL directory):
    uv run python scripts/summarize_height_curriculum.py logs/height-curriculum-YYYYMMDD/artifacts/logs/height-curriculum

Columns: block, top height, frontier successes per start (development seeds
20–22, out of 72), impacts / falls / side exits / timeouts summed over the
frontier, regression results (successes out of 72, P = passes the gate) and
whether the policy advanced. Crabbing per stage comes from
``analyze_course_drift.py`` on the frontier evaluation folders.
"""
import argparse
import json
from pathlib import Path

from analyze_course_drift import drift


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
                   advanced=entry['advanced'], frontier_folder=directory.name)
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
                  f"{r['timeouts']:>8}  {crab:>8}  {reg:<20}  {str(r['confirmed']):>9} {str(r['advanced']):>8}")
