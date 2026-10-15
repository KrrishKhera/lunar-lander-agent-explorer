import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import threading
import tempfile
from pathlib import Path
import gymnasium as gym
import numpy as np
from .policy import DQNPolicy

ENV_ID = 'LunarLander-v3'
ACTIONS = ['Coast', 'Left engine', 'Main engine', 'Right engine']
# Pygame rendering has process-global state; serialize runs across Streamlit sessions.
SIMULATION_LOCK = threading.Lock()

def run_episode(seed, agent='DQN', record=False, trace=False):
    if not isinstance(seed, int) or not 0 <= seed <= 1_000_000:
        raise ValueError('Seed must be an integer from 0 to 1,000,000.')
    if agent not in ('DQN','Random'): raise ValueError('Unknown agent.')
    with SIMULATION_LOCK:
        return _run(seed, agent, record, trace)

def _run(seed, agent, record, trace):
    env = gym.make(ENV_ID, render_mode='rgb_array' if record else None, max_episode_steps=1000)
    policy = DQNPolicy() if agent == 'DQN' else None
    rng = np.random.default_rng(seed + 10_000_000)
    total, counts, rows = 0.0, [0]*4, []
    video = None
    import imageio.v2 as imageio
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / 'episode.mp4'
        writer = None
        try:
            state, _ = env.reset(seed=seed)
            if record:
                writer = imageio.get_writer(str(video_path), fps=25, codec='libx264', macro_block_size=1)
                writer.append_data(env.render())
            for step in range(1,1001):
                action = policy.action(state) if policy else int(rng.integers(4))
                next_state, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                counts[action] += 1
                if trace:
                    rows.append({'step':step,'reward':float(reward),'total_reward':total,
                                 'action':ACTIONS[action], 'x':float(next_state[0]),
                                 'y':float(next_state[1]),'angle':float(next_state[4])})
                if record and (step % 2 == 0 or terminated or truncated):
                    writer.append_data(env.render())
                state = next_state
                if terminated or truncated: break
            # Report observed environment endings separately from the reward threshold.
            raw = env.unwrapped
            if raw.game_over: outcome = 'Crash'
            elif abs(float(state[0])) >= 1: outcome = 'Out of bounds'
            elif terminated and not raw.lander.awake: outcome = 'Landed / at rest'
            elif truncated: outcome = 'Time limit'
            else: outcome = 'Terminated'
            result = {'agent':agent, 'seed':seed, 'reward':round(total,6), 'steps':step,
                      'score_ge_200':total >= 200, 'outcome':outcome,
                      'terminated':bool(terminated),'truncated':bool(truncated),
                      'action_counts':counts}
        finally:
            if writer is not None: writer.close()
            env.close()
        if record: video = video_path.read_bytes()
    return result, rows, video
