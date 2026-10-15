"""Load and check the fixed Gymnasium v3 benchmark artifacts."""
import csv
import hashlib
import json
from pathlib import Path

from .policy import ROOT


def load_benchmark(root=ROOT):
    root = Path(root)
    summary = json.loads((root / 'results/summary.json').read_text())
    with (root / 'results/episodes.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    digest = hashlib.sha256((root / 'models/dqn_weights.npz').read_bytes()).hexdigest()
    if digest != summary['model_sha256']:
        raise ValueError('Saved model differs from the evaluated checkpoint.')
    seeds = summary['seeds']
    if len(rows) != 2 * len(seeds) or len(seeds) != 50:
        raise ValueError('Benchmark CSV is incomplete.')
    for agent in ('DQN', 'Random'):
        subset = [row for row in rows if row['agent'] == agent]
        if [int(row['seed']) for row in subset] != seeds:
            raise ValueError(f'{agent} seed rows differ from benchmark metadata.')
    for row in rows:
        row['seed'] = int(row['seed'])
        row['reward'] = float(row['reward'])
        row['steps'] = int(row['steps'])
        row['score_ge_200'] = row['score_ge_200'] == 'True'
    return summary, rows
