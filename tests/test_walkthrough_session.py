"""Selection and safety rules use all attempts, never only landings."""
import copy
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("session", Path(__file__).parents[1]/"scripts/run_walkthrough_session.py")
session = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session)


def candidate(success=0, capped_mean=2.25, impact=.75, landing=.86):
    m = dict(success=success,capped_mean=capped_mean,impact=impact,landing=landing)
    return dict(m,by_seed={str(s):m.copy() for s in (20,21,22)})


def test_selection_order_and_ties():
    source = candidate()
    assert session.rank(candidate(success=1/72,capped_mean=0)) > session.rank(source)
    assert session.rank(candidate(capped_mean=2.3,impact=1)) > session.rank(source)
    assert session.rank(candidate(impact=.7,landing=0)) > session.rank(source)
    assert session.rank(candidate(landing=.9)) > session.rank(source)
    assert not session.rank(copy.deepcopy(source)) > session.rank(source)


def test_separate_consecutive_checks_and_thresholds():
    source = candidate()
    walking = session.breaches(candidate(capped_mean=1.75),source,source)
    impact = session.breaches(candidate(impact=.9),source,source)
    assert walking == dict(strict=False,walking=True,impact=False)
    assert impact == dict(strict=False,walking=False,impact=True)
    streak = session.update_streaks({},walking)
    assert max(session.update_streaks(streak,impact).values()) == 1
    assert session.update_streaks(streak,walking)['walking'] == 2
    c = candidate(capped_mean=1.75)
    c['by_seed']['20']['capped_mean'] = 2.25
    c['by_seed']['21']['capped_mean'] = 2.25
    assert not session.breaches(c,source,source)['walking']


def test_all_attempt_denominators():
    rows = [dict(seed=s, prefix=dict(clean_prefix_blocks=5 if i==0 else 0),
                 sustained=dict(success=False,milestone_time_s=1 if i==0 else None,reasons=[]),
                 termination='time_out') for s in (20,21,22) for i in range(24)]
    score = session.score_rows(rows,'source')
    assert score['capped_mean'] == 5/24
    assert score['landing'] == 1/24
    assert score['histogram'] == [69,0,0,0,0,3]


def test_numerical_stop_patterns():
    spec = importlib.util.spec_from_file_location("numeric_watch", Path(__file__).parents[1]/"scripts/watch_walkthrough_numerics.py")
    watch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watch)
    for line in ('Mean value loss: nan', 'Mean entropy: inf', '[NanGuard] NaN/Inf found in envs: [0]'):
        assert watch.nonfinite(line)
    assert not watch.nonfinite('Mean value loss: 1.2; Mean noise std: 0.3')
