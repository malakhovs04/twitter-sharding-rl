"""Matched budgets and representations; no holdout file paths in training."""
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
import json,time,hashlib
import numpy as np
import pandas as pd
from agents import MLP,double_targets
from .prepare import load
from .environment import Env,calibrate,rollout,summary,state_key,encode

ROOT=Path('v2_results')

def table_policy(q):return lambda s:int(np.argmax(q.get(state_key(s),np.array([-1.,0.,-1.]))))-1

def neural_policy(net,matched):return lambda s:int(net.forward(encode(s,matched)).argmax())-1

def threshold(up,down):
    return lambda s: 1 if s[1]>up or s[2]>.01 else (-1 if s[1]<down else 0)

def target_policy(n):return lambda s:int(np.sign(n-(round(s[0]*6)+2)))

def validation(data,c,policy,protocol):
    return [summary(rollout(data,c,*w,policy)) for w in protocol['development_validation_windows']]

def fit(data,c,hp,seed,kind,protocol):
    rng=np.random.default_rng(seed);q={};matched=kind=='dqn_matched'
    net=MLP(32,seed);target=MLP(32,seed);target.copy_from(net)
    cap=12000;xs=np.zeros((cap,10));ns=xs.copy();aa=np.zeros(cap,dtype=int);rr=np.zeros(cap);dd=np.zeros(cap)
    steps=0;history=[];alpha=.05 if hp['rate']=='low' else .2;lr=.0003 if hp['rate']=='low' else .001
    started=time.monotonic()
    for ep in range(protocol['episodes']):
        # Schedule identical across methods/seeds, all three train days used.
        window=protocol['development_training_windows'][ep%3];env=Env(data,c,*window);s=env.reset();done=False;cost=0
        epsilon=max(.05,1-ep/(protocol['episodes']*.8))
        while not done:
            x=encode(s,matched)
            if rng.random()<epsilon:a=int(rng.integers(3))
            elif kind=='q':a=int(q.get(state_key(s),np.array([-1.,0.,-1.])).argmax())
            else:a=int(net.forward(x).argmax())
            nxt,r,done,row=env.step(a-1);cost+=row['cost']
            if kind=='q':
                k=state_key(s);nk=state_key(nxt)
                values=q.setdefault(k,np.zeros(3));future=q.get(nk,np.zeros(3)).max()
                y=r+(0 if done else hp['gamma']*future);values[a]+=alpha*(y-values[a])
            else:
                i=steps%cap;xs[i]=x;ns[i]=encode(nxt,matched);aa[i]=a;rr[i]=r;dd[i]=done
                if steps>=255 and steps%4==0:
                    idx=rng.integers(0,min(steps+1,cap),64)
                    y=double_targets(rr[idx],dd[idx],net.forward(ns[idx]),target.forward(ns[idx]),hp['gamma'])
                    net.update(xs[idx],aa[idx],y,lr)
                if (steps+1)%256==0:target.copy_from(net)
            s=nxt;steps+=1
        history.append(dict(episode=ep+1,training_cost=cost))
    return q if kind=='q' else net,history,dict(seconds=time.monotonic()-started,transitions=steps)

def save_table(q,path):
    keys=list(q);np.savez_compressed(path,keys=np.array(keys,dtype=np.int16),values=np.array([q[k] for k in keys]))
def load_table(path):
    with np.load(path) as f:return {tuple(k):v.copy() for k,v in zip(f['keys'],f['values'])}

def main():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'frozen.json').exists():raise RuntimeError('Frozen run exists; use a new experiment directory, no overwrite')
    protocol=json.loads(Path('v2/protocol.json').read_text());data=load('data/cluster7');c=calibrate(data)
    (ROOT/'config.json').write_text(json.dumps(asdict(c),indent=2))
    rows=[];selected={}
    for kind in ('q','dqn_matched','dqn_continuous'):
        for i,hp in enumerate(protocol['rl_candidates_each']):
            for seed in protocol['seeds']:
                model,history,budget=fit(data,c,hp,seed,kind,protocol);name=f'{kind}_{i}_s{seed}'
                if kind=='q':save_table(model,ROOT/(name+'.npz'));policy=table_policy(model)
                else:model.save(ROOT/(name+'.npz'));policy=neural_policy(model,kind=='dqn_matched')
                val=validation(data,c,policy,protocol)
                for fold,row in enumerate(val):rows.append(dict(kind=kind,candidate=i,seed=seed,fold=fold,**row,**budget))
                pd.DataFrame(history).to_csv(ROOT/(name+'_train.csv'),index=False)
                pd.DataFrame(rows).to_csv(ROOT/'validation.csv',index=False)
                print(name,'validation cost',np.mean([r['cost'] for r in val]),'seconds',budget['seconds'],flush=True)
        frame=pd.DataFrame(rows);part=frame[frame.kind==kind];best=int(part.groupby('candidate').cost.mean().idxmin())
        # Seed 7 predeclared for deployment; other independent initializations are reported, not cherry-picked.
        selected[kind]=dict(candidate=best,seed=7,path=f'{kind}_{best}_s7.npz',validation_cost=float(part[part.candidate==best].cost.mean()))
    baselines=[];started=time.monotonic()
    for up in protocol['threshold_grid']['up']:
        for down in protocol['threshold_grid']['down']:
            vals=validation(data,c,threshold(up,down),protocol)
            baselines.append(dict(up=up,down=down,validation_cost=float(np.mean([r['cost'] for r in vals]))))
    f=pd.DataFrame(baselines);f.to_csv(ROOT/'threshold_validation.csv',index=False);b=f.loc[f.validation_cost.idxmin()].to_dict()
    selected['threshold']=b;selected['threshold']['search_seconds']=time.monotonic()-started
    targets=[]
    for n in protocol['fixed_targets']:
        v=validation(data,c,target_policy(n),protocol);targets.append(dict(target=n,validation_cost=float(np.mean([r['cost'] for r in v]))))
    f=pd.DataFrame(targets);f.to_csv(ROOT/'target_validation.csv',index=False);selected['fixed_target']=f.loc[f.validation_cost.idxmin()].to_dict()
    (ROOT/'selection.json').write_text(json.dumps(selected,indent=2))
    files=list(Path('v2').glob('*.py'))+[Path('v2/protocol.json'),Path('agents.py'),Path('data_pipeline.py'),ROOT/'selection.json',ROOT/'config.json']
    files+=list(ROOT.glob('*.npz'))
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'frozen.json').write_text(json.dumps(dict(files=frozen,holdout_opened=False,note='Frozen before any holdout download/evaluation'),indent=2))
    print('FROZEN',json.dumps(selected),flush=True)
if __name__=='__main__':main()
