"""Paired seeded evaluation, kept separate from the historical training result."""
import argparse, csv, hashlib, importlib.metadata, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from lander.policy import ROOT
from lander.simulation import ENV_ID, run_episode

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=50)
    parser.add_argument('--start-seed',type=int,default=1000)
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    if not 2 <= args.episodes <= 1000: parser.error('Use 2–1000 episodes.')
    if not 0 <= args.start_seed <= 1_000_000-args.episodes: parser.error('Seed range invalid.')
    rows=[]
    for agent in ['DQN','Random']:
        for seed in range(args.start_seed,args.start_seed+args.episodes):
            result,_,_=run_episode(seed,agent)
            result.pop('action_counts');rows.append(result)
        print(f'{agent}: {args.episodes} episodes complete',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'episodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    stats={}
    for agent in ['DQN','Random']:
        subset=[r for r in rows if r['agent']==agent]
        rewards=np.array([r['reward'] for r in subset])
        stats[agent]={'episodes':len(subset),'mean_reward':float(rewards.mean()),
                     'std_reward':float(rewards.std(ddof=1)), 'median_reward':float(np.median(rewards)),
                     'min_reward':float(rewards.min()),'max_reward':float(rewards.max()),
                     'score_ge_200_rate':float(np.mean(rewards>=200)),
                     'mean_steps':float(np.mean([r['steps'] for r in subset]))}
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'environment':ENV_ID,
             'seeds':list(range(args.start_seed,args.start_seed+args.episodes)),
             'policy':'Greedy DQN (epsilon=0); uniform random baseline',
             'max_episode_steps':1000,'wind':False,'gravity':-10,
             'model_sha256':hashlib.sha256((ROOT/'models/dqn_weights.npz').read_bytes()).hexdigest(),
             'versions':{p:importlib.metadata.version(p) for p in ['gymnasium','numpy','box2d-py']},
             'python':platform.python_version(),'statistics':stats,
             'notes':['Same initial seed range for both policies; a new environment per episode.',
                      'Seeds were selected for this evaluation, not tuned. Historical training seeds are unknown.',
                      'Score >=200 is a reward threshold, not a physical landing-success label.',
                      'Transferred v2 checkpoint evaluated in v3; not a reproduction of the training run.']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
