"""Verification only: reproduce saved frozen-policy traces, never select or overwrite."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from compact_data import CompactEnv,load_compact
from v2.environment import calibrate
from v2.evaluate import policies
r=Path('v2_results');h=r/'holdout'
for path,digest in json.loads((r/'frozen.json').read_text())['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
selection=json.loads((r/'selection.json').read_text());count=0
for cluster in (10,26,45):
    d=load_compact(f'data/compact_cluster{cluster}');c=calibrate(d,0,96)
    for name,policy in policies(selection,True).items():
        env=CompactEnv(d,c,96,672);s=env.reset();done=False;rows=[]
        while not done:s,_,done,row=env.step(int(policy(s)));rows.append(row)
        actual=pd.DataFrame(rows);expected=pd.read_csv(h/f'cluster{cluster}_{name}.csv')
        np.testing.assert_allclose(actual.to_numpy(),expected.to_numpy(),rtol=1e-9,atol=1e-6);count+=1
print(f'Verified {count} complete policy traces, 576 slots each, and all frozen file hashes.')
