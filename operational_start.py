"""Local deployment experiment: causal pre-provisioning, unchanged safety-DQN weights."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from simulator import Env,Config,SPLITS,load_data,summarize,threshold
from agents import MLP,dqn_policy

def initial_shards(data,c,start,margin,history=4):
    if not 0<margin<=1:raise ValueError('margin must lie in (0, 1]')
    # Only requests strictly before deployment; never current-slot demand.
    past=data['loads'][max(0,start-history):start]
    if not len(past):return 8
    for n in range(2,9):
        if float(past[:,n-2,:n].max())<=margin*c.capacity:return n
    return 8

def evaluate_policy(data,c,split,policy,initial,prepared=True):
    start,end=SPLITS[split]
    preparation=[];history=None
    if prepared:
        warm=Env(data,c,start-6,start,initial=2)
        while warm.t<start:
            action=1 if warm.n<initial else 0
            _,_,_,row=warm.step(action);preparation.append(row)
        history=warm.history[-4:]
        assert warm.n==initial
    env=Env(data,c,start,end,initial=initial);s=env.reset()
    if history is not None:env.history=history;s=env.observe()
    done=False;rows=[]
    while not done:s,r,done,row=env.step(int(policy(s)));rows.append(row)
    trace=pd.DataFrame(rows);stats=summarize(trace)
    prep=pd.DataFrame(preparation)
    stats.update(initial_shards=initial,preparation_minutes=90 if prepared else 0,preparation_cost=float(prep.cost.sum()) if len(prep) else 0.,preparation_overloaded_slots=int((prep.excess>0).sum()) if len(prep) else 0,preparation_excess_requests=int(prep.excess.sum()) if len(prep) else 0,total_with_preparation=stats['cost']+(float(prep.cost.sum()) if len(prep) else 0.))
    return trace,stats

def study(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    data=load_data(root/'data/processed.npz');c=Config(**json.loads((root/'results/config.json').read_text()))
    selected=json.loads((root/'results/safety/selection.json').read_text());weight=root/'results/safety'/(selected['name']+'.npz')
    net=MLP.load(weight);policy=dqn_policy(net)
    protocol={'purpose':'Causal initial reserve only, no new training and no change to network weights or capacity','candidate_margins':[1.,.9,.8,.7],'history_slots':4,'decision_lead_slots':6,'selection':'validation overloaded slots, excess requests, total cost including preparation estimate','weights_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'test_note':'Previously observed test, repeated diagnostic evaluation','preparation_note':'Decision uses only requests before t=start-6. Replay six real pre-start intervals, expanding at most one shard per step; charge actual simulated server, overload and accumulated repeated migration costs. Publish pre-start overload separately; zero post-start does not mean zero during warmup.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for margin in protocol['candidate_margins']:
        n=initial_shards(data,c,384-6,margin);_,s=evaluate_policy(data,c,'validation',policy,n);rows.append(dict(margin=margin,**s))
    pd.DataFrame(rows).to_csv(out/'validation.csv',index=False)
    chosen=min(rows,key=lambda x:(x['overloaded_slots'],x['excess_requests'],x['total_with_preparation']))
    (out/'selection.json').write_text(json.dumps(chosen,indent=2))
    n=initial_shards(data,c,480-6,chosen['margin'])
    rows=[]
    base=json.loads((root/'results/baseline_selection.json').read_text())
    cases=[('safety_dqn_cold',policy,2),('safety_dqn_reserve',policy,n),('threshold_same_reserve',threshold(base['up'],base['down']),n)]
    for fixed in (6,7,8):cases.append((f'fixed_{fixed}',lambda s:0,fixed))
    for name,p,initial in cases:
        trace,s=evaluate_policy(data,c,'test',p,initial,prepared=name!='safety_dqn_cold');trace.to_csv(out/(name+'.csv'),index=False);rows.append(dict(strategy=name,**s))
    summary=pd.DataFrame(rows).set_index('strategy');summary.to_csv(out/'comparison.csv')
    print('Validation-chosen margin:',chosen['margin'],'test initial shards:',n);print(summary.to_string())
    return summary
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',default=str(Path(__file__).resolve().parent));parser.add_argument('--output',default='results/operational_start');a=parser.parse_args();study(a.project_root,a.output)
