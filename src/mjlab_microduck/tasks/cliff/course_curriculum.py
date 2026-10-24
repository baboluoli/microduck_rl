"""Evaluation-driven pilot decisions. No iteration-based height advancement.

Each condition is one height/distance pair with exactly 24 first episodes on
three seeds. Keep development and confirmation results separate at the caller.
"""
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path


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
    increment_mm: float = 2.5
    resolution_mm: float = .5
    start_range: tuple[float, float] = (.45, .65)
    history: list = field(default_factory=list)
    baselines: dict = field(default_factory=dict)
    streaks: dict = field(default_factory=dict)

    def conditions(self):
        return {f'{h:g}/{d}': None for h in (20 + i*2.5 for i in range(round((self.upper_mm-20)/2.5)+1))
                for d in (45, 55, 65)}

    def advance(self, development, confirmation, checkpoint):
        required = self.conditions()
        if set(development) != set(required) or set(confirmation) != set(required):
            raise ValueError('Incomplete height/distance gate')
        if not all(r.passes for batch in (development, confirmation) for r in batch.values()):
            return False
        # Baselines use development seeds 20–22, matching subsequent checks.
        # Confirmation seeds 40–42 gate advancement but are never mixed into
        # a per-seed regression comparison against different seeds.
        for key, result in development.items():
            self.baselines.setdefault(key, list(result.successes))
        if self.upper_mm >= 30:
            return False
        self.history.append(dict(upper_mm=self.upper_mm, checkpoint=checkpoint,
                                 development={k:asdict(v) for k,v in development.items()},
                                 confirmation={k:asdict(v) for k,v in confirmation.items()}))
        self.upper_mm += self.increment_mm
        return True

    def regression(self, results):
        if not self.baselines.keys() <= results.keys():
            raise ValueError('A completed check must cover every mastered condition')
        for key, base in self.baselines.items():
            current = results[key].successes
            breach = (sum(base)-sum(current) >= .15*72
                      and sum(b-c >= .10*24 for b,c in zip(base,current)) >= 2)
            self.streaks[key] = self.streaks.get(key, 0)+1 if breach else 0
        return any(n >= 2 for n in self.streaks.values())

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(asdict(self), indent=2)+'\n')
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
