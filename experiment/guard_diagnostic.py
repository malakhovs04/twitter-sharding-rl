"""POST-HOLDOUT exploratory safeguard, never part of frozen primary comparison.
Reject new migration if recent writes consume >=80% of bandwidth, or estimated
snapshot+dirty-log completion exceeds eight slots. No future workloads used.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from compact_data import load_compact as load, CompactEnv as Env
from v2.environment import calibrate,summary
from v2.evaluate import policies

def admit(env,action):
    if not action or env.target!=env.n:return action,False,0.,0.
    n=int(np.clip(env.n+action,2,8))
    if n==env.n:return action,False,0.,0.
    mask=env.d['mapping'][env.n-2]!=env.d['mapping'][n-2]
    # Only completed slots and boundary inventory, never writes[t].
    start=max(0,env.t-4)
    dirty=float(env.d['writes'][start:env.t,mask].sum(axis=1).mean()) if env.t>start else 0.
    snapshot=float(env.d['inventory'][env.t,mask].sum())
    duration=snapshot/max(env.c.bandwidth-dirty,1e-12)
    reject=dirty>=.8*env.c.bandwidth or duration>8
    return 0 if reject else action,bool(reject),dirty,duration

def run(data,c,policy):
    e=Env(data,c,96,672);s=e.reset();done=False;rows=[]
    while not done:
        requested=int(policy(s));a,rejected,rate,duration=admit(e,requested)
        s,_,done,row=e.step(a);rows.append(dict(**row,requested_action=requested,admission_rejected=int(rejected),estimated_dirty_bytes=rate,estimated_slots=duration))
    return rows

def main():
    r=Path('v2_results');out=r/'post_holdout_guard';out.mkdir(exist_ok=True)
    if (out/'comparison.csv').exists():raise RuntimeError('Diagnostic already run')
    (out/'protocol.json').write_text(json.dumps(dict(status='EXPLORATORY POST-HOLDOUT, created after observing cluster26 stalled migration; no independent performance claim',dirty_bandwidth_limit=.8,max_estimated_slots=8,past_slots=4,abort_active=False,limitations='Past rate does not bound future rate; no guarantee of completion or zero queue; may conservatively reject helpful moves'),indent=2))
    selection=json.loads((r/'selection.json').read_text());rows=[]
    for cluster in (10,26,45):
        d=load(f'data/compact_cluster{cluster}');c=calibrate(d,0,96)
        for name,policy in policies(selection).items():
            trace=run(d,c,policy);pd.DataFrame(trace).to_csv(out/f'cluster{cluster}_{name}.csv',index=False)
            rows.append(dict(cluster=cluster,strategy=name,rejections=sum(x['admission_rejected'] for x in trace),**summary(trace)))
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
