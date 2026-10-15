from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

class DQNPolicy:
    """Inference-only copy of the saved Dense/ReLU Keras network, using NumPy."""
    def __init__(self, path=ROOT / 'models/dqn_weights.npz'):
        with np.load(path, allow_pickle=False) as data:
            self.layers = [(data[f'w{i}'].copy(), data[f'b{i}'].copy()) for i in range(3)]
        expected = [(8,64),(64,64),(64,4)]
        for (w,b), shape in zip(self.layers, expected):
            if w.shape != shape or b.shape != (shape[1],):
                raise ValueError('Model architecture does not match 8 → 64 → 64 → 4.')
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('Model contains non-finite weights.')

    def q_values(self, states):
        x = np.asarray(states, dtype=np.float32)
        if x.ndim not in (1,2) or x.shape[-1] != 8 or not np.isfinite(x).all():
            raise ValueError('Expected finite observations with eight values.')
        for i, (w,b) in enumerate(self.layers):
            x = x @ w + b
            if i < 2: x = np.maximum(x, 0)
        return x

    def action(self, state):
        return int(np.argmax(self.q_values(state)))
