"""Re-score saved first episodes for a landing plus one metre of follow-through.

This is an offline, post-hoc metric: it changes neither training nor historical
scores. The saved lower_floor flag comes from found contacts whose positions
were checked against the real lower floor (not the old landing hold).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def classify(trace, columns, distance_m=1.0, dt=.02):
    """First lower-floor contact anchors net world-X distance, once per episode.

    Tilt invalidity starts at that contact, including the contact sample.
    Head/trunk impacts, nonfinite evidence and boundary departures veto success
    from reset. Check failure before success so a failure on the goal sample
    cannot be hidden. Airborne phases do not restart the distance measurement.
    A later failure is reported separately: the proposed task would have ended
    at the earlier success sample.
    """
    if distance_m <= 0 or not np.isfinite(distance_m):
        raise ValueError('distance_m must be finite and positive')
    c = dict(zip(columns, trace.T))
    required = {'t', 'x', 'y', 'tilt_deg', 'head_force', 'trunk_force',
                'bad_impact', 'lower_floor'}
    if not required <= c.keys() or len(trace) == 0:
        raise ValueError('Missing evidence or empty first episode')
    ages = np.rint(c['t'] / dt).astype(int)
    if not np.array_equal(ages, np.arange(1, len(trace)+1)):
        raise ValueError('Expected contiguous first-episode samples')
    landing = success = first_failure = None
    bad = False
    pre_goal_events, later_events = set(), set()
    max_clean_progress = 0.
    for k, sample in enumerate(trace):
        x, y = float(c['x'][k]), float(c['y'][k])
        finite = bool(np.isfinite(sample).all())
        if landing is None and finite and c['lower_floor'][k] > .5:
            landing = dict(step=int(ages[k]), time_s=float(ages[k]*dt), root_x_m=x)
        reasons = []
        if not finite:
            reasons.append('nonfinite')
        if (c['bad_impact'][k] > .5 or c['head_force'][k] > 15
                or c['trunk_force'][k] > 20):
            reasons.append('impact')
        if x < -.75 or x > 3.8 or abs(y) > 1.15:
            reasons.append('boundary')
        if landing is not None and c['tilt_deg'][k] >= 45 - 1e-5:
            reasons.append('tilt')
        if success is None:
            pre_goal_events.update(reasons)
        else:
            later_events.update(reasons)
        if reasons:
            bad = True
            if first_failure is None:
                first_failure = dict(step=int(ages[k]), time_s=float(ages[k]*dt),
                                     reasons=reasons)
        if landing is not None and not bad:
            progress = x - landing['root_x_m']
            max_clean_progress = max(max_clean_progress, progress)
            # One micrometre accommodates float32 position subtraction only.
            if success is None and progress >= distance_m - 1e-6:
                success = dict(step=int(ages[k]), time_s=float(ages[k]*dt),
                               root_x_m=x, progress_m=progress)
    failure_before_goal = (first_failure is not None and
                           (success is None or first_failure['step'] <= success['step']))
    return dict(success=success is not None, first_lower_contact=landing,
                completion=success, first_failure=first_failure,
                failure_before_goal=bool(failure_before_goal),
                later_failure_after_success=bool(success is not None and first_failure is not None),
                pre_goal_events=sorted(pre_goal_events),
                later_events_after_success=sorted(later_events),
                max_clean_progress_m=max_clean_progress,
                observed_duration_s=float(ages[-1]*dt),
                no_recorded_lower_contact=landing is None)


def summarize(rows):
    n = len(rows)
    times = [r['completion']['time_s'] for r in rows if r['success']]
    def early(reason):
        return sum(r['failure_before_goal'] and reason in r['first_failure']['reasons']
                   for r in rows)
    return dict(attempts=n, successes=sum(r['success'] for r in rows),
                success_rate=sum(r['success'] for r in rows)/n,
                recorded_lower_contacts=sum(not r['no_recorded_lower_contact'] for r in rows),
                first_disqualification_before_goal={name:early(name) for name in ('impact','tilt','boundary','nonfinite')},
                observed_events_before_goal_or_end={
                    name:sum(name in r['pre_goal_events'] for r in rows)
                    for name in ('impact','tilt','boundary','nonfinite')},
                no_recorded_lower_contact=sum(r['no_recorded_lower_contact'] for r in rows),
                insufficient_clean_distance=sum(not r['success'] and not r['failure_before_goal']
                                                and not r['no_recorded_lower_contact'] for r in rows),
                successes_with_later_old_protocol_invalidity=sum(r['later_failure_after_success'] for r in rows),
                successes_with_later_physical_invalidity=sum(
                    bool(set(r['later_events_after_success']) & {'impact','tilt','nonfinite'}) for r in rows),
                successes_with_later_boundary=sum('boundary' in r['later_events_after_success'] for r in rows),
                successes_later_terminated_by_impact=sum(r['success'] and 'cliff_failed_landing' in r['termination'] for r in rows),
                full_trace_impact_terminations=sum('cliff_failed_landing' in r['termination'] for r in rows),
                full_trace_boundary_terminations=sum('boundary' in r['termination'] for r in rows),
                success_time_s=None if not times else dict(mean=float(np.mean(times)),
                    min=min(times), median=float(np.median(times)), max=max(times)))


def analyze(source, distance_m=1.):
    manifest = json.loads((source/'manifest.json').read_text())
    episodes = json.loads((source/'episodes.json').read_text())
    assert manifest['seeds'] == [20,21,22] and manifest['num_envs'] == 24
    assert manifest['approach_m'] == .55
    fingerprints = {}
    def fingerprint(path):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        fingerprints[path.name] = sha
        return sha
    fingerprint(source/'manifest.json')
    fingerprint(source/'episodes.json')
    rows = []
    for seed in manifest['seeds']:
        paths = list(source.glob(f'*-s{seed}-d0p55.npz'))
        assert len(paths) == 1
        fingerprint(paths[0])
        seed_rows = sorted((r for r in episodes if r['seed']==seed), key=lambda r:r['env'])
        assert [r['env'] for r in seed_rows] == list(range(24))
        with np.load(paths[0], allow_pickle=False) as z:
            columns = list(z['columns'])
            assert columns == manifest['columns']
            for row in seed_rows:
                assert row['height_cm'] == 2 and row['approach_m'] == .55
                assert 0 < row['steps'] <= z['values'].shape[0]
                score = classify(z['values'][:row['steps'],row['env']], columns, distance_m)
                rows.append(dict(seed=seed, env=row['env'], termination=row['termination'], **score))
    assert len(rows) == 72
    return dict(source=str(source), checkpoint_sha256=manifest['sha256'],
                device=manifest['device'], distance_m=distance_m,
                input_sha256=fingerprints, pooled=summarize(rows),
                by_seed={str(s):summarize([r for r in rows if r['seed']==s]) for s in manifest['seeds']},
                episodes=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.session.resolve()):
        parser.error('Write the new analysis outside the immutable session artifacts')
    results = {name:analyze(args.session/name) for name in ('reference','eval-model_100','eval-model_249')}
    output = dict(metric='first genuine lower-floor foot contact + 1.0 m net world-X travel',
                  post_hoc=True, training_changed=False, heldout_used=False,
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  checkpoints=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k:dict(pooled=v['pooled'],by_seed=v['by_seed']) for k,v in results.items()},indent=2))


if __name__ == '__main__':
    main()
