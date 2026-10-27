"""Warm-start DQN study: prioritize overload frequency, then amount, then common cost."""
from pathlib import Path
from dataclasses import asdict
import itertools,json,hashlib
import numpy as np
import pandas as pd
from agents import MLP,dqn_policy,double_targets
from simulator import Env,Config,load_data,SPLITS,rollout,summarize,baselines
ROOT=Path(__file__).resolve().parent
# Run from the repository root; source data preparation is documented in README.
OUT=Path(__file__).resolve().parent/'results/safety'
def js(path,value):Path(path).write_text(json.dumps(value,indent=2))
def metric(trace):return (int((trace.excess>0).sum()),int(trace.excess.sum()),float(trace.cost.sum()))
def adjusted_reward(row,c,multiplier,event):
    return -(row['cost']+(multiplier-1)*c.overload_price*row['excess']/c.capacity+event*(row['excess']>0))/10

def fine_tune(data,c,initial,hp,seed=7,episodes=240):
    rng=np.random.default_rng(seed);net=MLP.load(initial);target=MLP.load(initial)
    cap=20000;xs=np.zeros((cap,10));ns=xs.copy();aa=np.zeros(cap,dtype=int);rr=np.zeros(cap);dd=rr.copy()
    count=0;step=0;history=[];best=None;bestp=None
    for ep in range(episodes):
        env=Env(data,c,*SPLITS['train']);s=env.reset();done=False;loss=[]
        eps=max(.03,.4*(1-ep/(episodes*.85)))
        while not done:
            a=int(rng.integers(3)) if rng.random()<eps else int(net.forward(s).argmax())
            nxt,_,done,row=env.step(a-1);i=count%cap
            xs[i]=s;ns[i]=nxt;aa[i]=a;rr[i]=adjusted_reward(row,c,hp['multiplier'],hp['event']);dd[i]=done
            count+=1;step+=1;s=nxt
            if count>=256 and step%4==0:
                idx=rng.integers(0,min(cap,count),64)
                y=double_targets(rr[idx],dd[idx],net.forward(ns[idx]),target.forward(ns[idx]),.9)
                loss.append(net.update(xs[idx],aa[idx],y,.0003))
            if step%256==0:target.copy_from(net)
        if (ep+1)%30==0 or ep==episodes-1:
            val=rollout(data,c,'validation',dqn_policy(net));key=metric(val)
            history.append(dict(episode=ep+1,overloaded_slots=key[0],excess_requests=key[1],cost=key[2],loss=float(np.mean(loss))))
            if best is None or key<best:best=key;bestp=[p.copy() for p in net.p]
    net.p=bestp;return net,history

def oracle_trace(data,c,split):
    """Perfect-future lexicographic lower bound under identical initial and +/-1 constraints."""
    start,end=SPLITS[split];values=[[(0,0,0.) for _ in range(7)] for _ in range(end-start+1)]
    acts=np.zeros((end-start,7),int)
    for t in range(end-1,start-1,-1):
        for i in range(7):
            candidates=[]
            for action in (-1,0,1):
                j=min(6,max(0,i+action));n=j+2;v=data['loads'][t,j,:n]
                excess=int(np.maximum(v-c.capacity,0).sum())
                cost=n*c.shard_price+c.overload_price*excess/c.capacity+c.migration_price*data['migrations'][t,i,j]/c.migration_scale
                later=values[t-start+1][j]
                candidates.append((int(excess>0)+later[0],excess+later[1],cost+later[2]))
            k=min(range(3),key=lambda k:candidates[k]);values[t-start][i]=candidates[k];acts[t-start,i]=k-1
    env=Env(data,c,start,end);env.reset();rows=[]
    while env.t<end:
        _,_,_,row=env.step(int(acts[env.t-start,env.n-2]));rows.append(row)
    return pd.DataFrame(rows)

def study():
    OUT.mkdir(parents=True,exist_ok=True);data=load_data(ROOT/'data/processed.npz');c=Config(**json.loads((ROOT/'results/config.json').read_text()))
    old=json.loads((ROOT/'results/dqn/selection.json').read_text());initial=ROOT/'results/dqn'/(old['name']+'.npz')
    protocol=dict(objective='lexicographic: overloaded slots, excess requests, original common cost',warm_start=str(initial),warm_start_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),multipliers=[3,10,30],event_penalties=[0,10],seeds=[7,42,123],episodes=240,checkpoint=30,learning_rate=.0003,gamma=.9,split=SPLITS,config=asdict(c),test_note='Previously seen test; repeated evaluation, no fresh independent holdout')
    js(OUT/'protocol.json',protocol)
    oracle=oracle_trace(data,c,'validation');oracle.to_csv(OUT/'validation_oracle.csv',index=False)
    rows=[]
    for multiplier,event in itertools.product((3,10,30),(0,10)):
        hp=dict(multiplier=multiplier,event=event)
        for seed in (7,42,123):
            name=f'm{multiplier}_e{event}_s{seed}';net,h=fine_tune(data,c,initial,hp,seed=seed)
            net.save(OUT/(name+'.npz'));pd.DataFrame(h).to_csv(OUT/(name+'_history.csv'),index=False)
            val=rollout(data,c,'validation',dqn_policy(net));row=dict(name=name,**hp,seed=seed,**summarize(val));rows.append(row)
            pd.DataFrame(rows).to_csv(OUT/'validation_search.csv',index=False)
            print(name,'validation',metric(val),flush=True)
    frame=pd.DataFrame(rows);means=frame.groupby(['multiplier','event'])[['overloaded_slots','excess_requests','cost']].mean()
    best_hp=min(means.index,key=lambda i:tuple(means.loc[i]))
    candidates=[r for r in rows if (r['multiplier'],r['event'])==best_hp]
    chosen=min(candidates,key=lambda r:(r['overloaded_slots'],r['excess_requests'],r['cost']))
    js(OUT/'selection.json',chosen)
    # All decisions frozen before repeated test comparison. Never compare changed reward scales as costs.
    traces={'original_dqn':rollout(data,c,'test',dqn_policy(MLP.load(initial))), 'safety_dqn':rollout(data,c,'test',dqn_policy(MLP.load(OUT/(chosen['name']+'.npz'))))}
    base=json.loads((ROOT/'results/baseline_selection.json').read_text())
    traces.update(baselines(data,c,'test',base['up'],base['down']))
    traces['future_oracle']=oracle_trace(data,c,'test')
    summary=pd.DataFrame({k:summarize(v) for k,v in traces.items()}).T
    summary.to_csv(OUT/'comparison.csv')
    for name,trace in traces.items():trace.to_csv(OUT/('test_'+name+'.csv'),index=False)
    seedrows=[]
    for row in candidates:
        trace=rollout(data,c,'test',dqn_policy(MLP.load(OUT/(row['name']+'.npz'))))
        seedrows.append(dict(seed=row['seed'],**summarize(trace)))
    pd.DataFrame(seedrows).to_csv(OUT/'test_seeds.csv',index=False)
    print('Selected',chosen['name']);print(summary.to_string());return summary
if __name__=='__main__':study()
