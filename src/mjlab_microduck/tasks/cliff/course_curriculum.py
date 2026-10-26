"""Evaluation-driven pilot decisions. No iteration-based height advancement.

Each condition is one height/distance pair with exactly 24 first episodes on
three seeds. Keep development and confirmation results separate at the caller.

The height climbs ``course.STAGES`` (2 cm up to 14 cm). Every check has a fixed
cost, however high the curriculum is:

- frontier: the current top height at starts 45/55/65 cm. It must pass on the
  development seeds (20–22) AND the confirmation seeds (40–42) to advance;
- regression: 2 cm at 55 cm, plus the previous stage at 55 cm, on the
  development seeds only. Failing these twice in a row means the policy is
  forgetting easier drops, and the session stops.
"""
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path

from .course import STAGES

DISTANCES = (45, 55, 65)
REGRESSION_DISTANCE = 55


def key(height_mm, distance_cm):
    """Condition name, e.g. ``'22.5/55'`` (height in mm / start in cm)."""
    return f'{height_mm:g}/{distance_cm}'


@dataclass
class Result:
    successes: tuple[int, int, int]
    impacts: int
    one_metre: int

    def __post_init__(self):
        if len(self.successes) != 3 or any(not 0 <= n <= 24 for n in self.successes):
            raise ValueError('Require three complete 24-attempt seed batches')
        if not 0 <= self.impacts <= 72 or not 0 <= self.one_metre <= 72:
            raise ValueError('Invalid all-attempt counts')

    @property
    def passes(self):
        return sum(self.successes) >= 65 and min(self.successes) >= 21 and self.impacts <= 3


def rank(results):
    """Only compare candidates at the same stage; strict > retains earlier ties."""
    return (sum(sum(r.successes) for r in results),
            -sum(r.impacts for r in results), sum(r.one_metre for r in results))


@dataclass
class Curriculum:
    upper_mm: float = 20
    harder_share: float = .75
    # Unused since the ladder follows course.STAGES; kept so that older
    # curriculum.json files (2.5 mm steps, 0.5 mm bins) still load.
    increment_mm: float = 2.5
    resolution_mm: float = .5
    start_range: tuple[float, float] = (.45, .65)
    history: list = field(default_factory=list)
    # Old success-drop regression rule (records only; no longer updated).
    baselines: dict = field(default_factory=dict)
    streaks: dict = field(default_factory=dict)
    # Current rule: consecutive failed checks per regression condition. Kept
    # apart from ``streaks`` so streaks under the old rule are not inherited.
    regression_streaks: dict = field(default_factory=dict)
    blocks_at_stage: int = 0

    @property
    def stage(self):
        return STAGES.index(self.upper_mm / 1000)

    @property
    def last_stage(self):
        return self.stage == len(STAGES) - 1

    def frontier(self):
        return [key(self.upper_mm, d) for d in DISTANCES]

    def regression_conditions(self):
        heights = [20] + ([STAGES[self.stage - 1] * 1000] if self.stage > 1 else [])
        return [key(h, REGRESSION_DISTANCE) for h in heights]

    def conditions(self):
        """Every condition of one development check, frontier first (3–5 total)."""
        return {k: None for k in self.frontier() + self.regression_conditions()}

    def advance(self, development, confirmation, checkpoint):
        """Raise the height one stage if the frontier passes on both seed sets.

        A pass at the last stage (14 cm) is recorded in ``history`` but cannot
        raise the height; the caller sees ``mastered`` become True.
        """
        required = set(self.frontier())
        if not required <= set(development) or set(confirmation) != required:
            raise ValueError('Incomplete height/distance gate')
        if not all(batch[k].passes for batch in (development, confirmation) for k in required):
            return False
        self.history.append(dict(upper_mm=self.upper_mm, checkpoint=checkpoint,
                                 blocks=self.blocks_at_stage,
                                 development={k:asdict(v) for k,v in development.items()},
                                 confirmation={k:asdict(v) for k,v in confirmation.items()}))
        if self.last_stage:
            return False
        self.upper_mm = round(STAGES[self.stage + 1] * 1000, 1)
        self.blocks_at_stage = 0
        return True

    def regression(self, results):
        """True when a regression condition has failed two checks in a row."""
        required = self.regression_conditions()
        if not set(required) <= results.keys():
            raise ValueError('A completed check must cover every regression condition')
        # The previous stage changes on advance; drop streaks that are no
        # longer checked so they cannot trigger a stop later.
        self.regression_streaks = {k: self.regression_streaks.get(k, 0) for k in required}
        for k in required:
            self.regression_streaks[k] = 0 if results[k].passes else self.regression_streaks[k] + 1
        return any(n >= 2 for n in self.regression_streaks.values())

    @property
    def mastered(self):
        """True once the last stage itself has passed its gate."""
        return self.last_stage and any(h['upper_mm'] == self.upper_mm for h in self.history)

    def stalled(self, max_blocks=5):
        return self.blocks_at_stage >= max_blocks

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(asdict(self), indent=2)+'\n')
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
