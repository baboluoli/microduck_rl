"""Zero the never-trained drop-height input of a checkpoint before the height ladder.

The oracle observation ends with ``[edge distance, drop height]``. A policy
trained only at 2 cm never saw the height change, so:
- its normalizer has mean 0.02 and variance 0 for that input;
- that input's first-layer weights never received a gradient and are still
  random. At taller drops they feed noise into every hidden unit.

This script sets those weight columns (``mlp.0.weight[:, i]``) to 0 in the
actor and the critic, so training starts from "ignore the height" and learns
how to use it. Everything else (normalizer, optimizer, ``iter``, ``infos``,
biases) is left untouched.

It refuses to guess: the height must be the LAST input column, its normalizer
mean must be about 0.02 and its variance about 0, and Adam's moments for those
columns must be negligible next to the trained columns (no real gradient
reached them, so no optimizer surgery is needed). Any mismatch raises.

On model_2000 (2026-10-25) the critic's moments were exactly 0. The actor's
were about 1e-11 against 8e-5 for the trained columns: float32 rounding of
``0.02 - mean``, not learning. Hence a relative tolerance, not exact zero.

Usage:
    uv run python scripts/prepare_height_input.py SOURCE.pt OUTPUT.pt
Writes OUTPUT.pt and OUTPUT.json (source sha, output sha, indices).
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

TRAINED_HEIGHT = 0.02  # The only drop height the source saw (metres).
NETWORKS = ('actor_state_dict', 'critic_state_dict')
# Height-column Adam moments / mean moments of the other columns, at most.
MOMENT_RATIO = 1e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def height_column(state_dict, name):
    """Index of the drop-height input, after checking it really is untrained."""
    weight = state_dict['mlp.0.weight']
    mean = state_dict['obs_normalizer._mean'].reshape(-1)
    var = state_dict['obs_normalizer._var'].reshape(-1)
    if not (weight.shape[1] == mean.numel() == var.numel()):
        raise ValueError(f'{name}: first layer has {weight.shape[1]} inputs, normalizer {mean.numel()}')
    i = weight.shape[1] - 1  # The oracle appends [edge distance, drop height]: height is last.
    if abs(float(mean[i]) - TRAINED_HEIGHT) > 1e-4:
        raise ValueError(f'{name}: column {i} mean {float(mean[i]):.5f}, expected {TRAINED_HEIGHT} (not the height?)')
    if float(var[i]) > 1e-8:
        raise ValueError(f'{name}: column {i} variance {float(var[i]):.3g}; the height input has been trained')
    return i


def adam_moments(optimizer_state, shape):
    """Adam state of the single parameter with this shape (first layers are unique)."""
    matches = [s for s in optimizer_state['state'].values() if tuple(s['exp_avg'].shape) == tuple(shape)]
    if len(matches) != 1:
        raise ValueError(f'Expected one optimizer entry of shape {tuple(shape)}, found {len(matches)}')
    return matches[0]


def zero_height_input(payload):
    """Zero the height column of both first layers in place. Returns the indices."""
    indices = {}
    for name in NETWORKS:
        state_dict = payload[name]
        i = height_column(state_dict, name)
        weight = state_dict['mlp.0.weight']
        moments = adam_moments(payload['optimizer_state_dict'], weight.shape)
        ratio = max(float(moments[k][:, i].abs().max() / moments[k][:, :i].abs().mean())
                    for k in ('exp_avg', 'exp_avg_sq'))
        if ratio > MOMENT_RATIO:
            raise ValueError(f'{name}: Adam moments of column {i} are {ratio:.3g} of the trained columns; '
                             f'it was trained')
        print(f'{name}: column {i}, mean |w| {float(weight[:, i].abs().mean()):.4f} -> 0; '
              f'Adam moments {ratio:.3g} of the trained columns, optimizer untouched')
        weight[:, i] = 0
        indices[name] = i
    return indices


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = torch.load(args.source, map_location='cpu', weights_only=False)
    indices = zero_height_input(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    record = dict(source=str(args.source), source_sha256=sha(args.source), output=str(args.output),
                  output_sha256=sha(args.output), zeroed_columns=indices, iter=payload.get('iter'))
    args.output.with_suffix('.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
