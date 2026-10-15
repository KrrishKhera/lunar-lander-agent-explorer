"""Extract Dense weights from the user's Keras H5 file without changing them."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from lander.policy import ROOT

source = ROOT/'models/original_dqn.h5'
with h5py.File(source,'r') as f:
    config=json.loads(f.attrs['model_config'])
    layers=[l for l in config['config']['layers'] if l['class_name']!='InputLayer']
    assert [l['class_name'] for l in layers]==['Dense']*3
    assert [l['config']['activation'] for l in layers]==['relu','relu','linear']
    weights={}
    for i,layer in enumerate(layers):
        name=layer['config']['name']; group=f['model_weights'][name]
        datasets={}
        group.visititems(lambda n,o: datasets.update({n:np.asarray(o)}) if isinstance(o,h5py.Dataset) else None)
        weights[f'w{i}']=next(v for k,v in datasets.items() if k.endswith('kernel:0'))
        weights[f'b{i}']=next(v for k,v in datasets.items() if k.endswith('bias:0'))
np.savez(ROOT/'models/dqn_weights.npz',**weights)
meta={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'architecture':[8,64,64,4],'activations':['relu','relu','linear'],
      'training_environment':'OpenAI Gym LunarLander-v2',
      'training_record':{'episode':455,'last_100_mean_reward':200.05},
      'source':'User-supplied Coursera coursework checkpoint; no retraining performed.'}
(ROOT/'models/model_info.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
