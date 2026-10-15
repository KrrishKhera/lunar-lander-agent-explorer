"""Verify lightweight predictions against TensorFlow loaded from the original H5."""
import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import tensorflow as tf
from lander.policy import ROOT,DQNPolicy
from lander.simulation import run_episode
import gymnasium as gym
original=tf.keras.models.load_model(ROOT/'models/original_dqn.h5',compile=False)
# Keras 3 interprets this old model's input metadata with an extra dimension.
# Rebuild the exact inspected Dense architecture, retaining all original weights.
reference=tf.keras.Sequential([tf.keras.Input(shape=(8,)),tf.keras.layers.Dense(64,activation='relu'),
 tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(4)])
reference.set_weights(original.get_weights())
x=np.random.default_rng(42).normal(size=(1000,8)).astype('float32')
policy=DQNPolicy()
a=reference(x,training=False).numpy();b=policy.q_values(x)
np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-4)
assert np.array_equal(a.argmax(1),b.argmax(1))
env=gym.make('LunarLander-v3',max_episode_steps=1000)
states=[]
try:
    state,_=env.reset(seed=1000)
    for _ in range(1000):
        states.append(state)
        state,_,terminated,truncated,_=env.step(policy.action(state))
        if terminated or truncated: break
finally:
    env.close()
real=np.asarray(states,dtype='float32')
real_reference=reference(real,training=False).numpy()
real_numpy=policy.q_values(real)
np.testing.assert_allclose(real_reference,real_numpy,rtol=1e-4,atol=1e-4)
assert np.array_equal(real_reference.argmax(1),real_numpy.argmax(1))
single_reference=reference(real[:1],training=False).numpy()[0]
single_numpy=policy.q_values(real[0])
np.testing.assert_allclose(single_reference,single_numpy,rtol=1e-4,atol=1e-4)
assert int(single_reference.argmax())==policy.action(real[0])
r,_,_=run_episode(1000)
r2,_,_=run_episode(1000)
assert r==r2,'Repeated seed differed'
info={'max_abs_q_error':float(np.max(np.abs(a-b))),'action_agreement':1.0,
      'random_observations':1000,'rtol':1e-4,'atol':1e-4,
      'real_observations':len(real),
      'real_max_abs_q_error':float(np.max(np.abs(real_reference-real_numpy))),
      'real_action_agreement':float(np.mean(real_reference.argmax(1)==real_numpy.argmax(1))),
      'single_observation_max_abs_q_error':float(np.max(np.abs(single_reference-single_numpy))),
      'reference':'TensorFlow Dense network rebuilt with weights from original H5 due to legacy input metadata',
      'episode':r,'repeat_seed_identical':True}
(ROOT/'results').mkdir(exist_ok=True)
(ROOT/'results/smoke_check.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info,indent=2))
