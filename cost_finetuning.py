"""Local cost fine-tuning: real train plus explicitly labelled demand-only stress scenarios."""
from pathlib import Path
import argparse,itertools,json,hashlib,time
import numpy as np
import pandas as pd
from agents import MLP,dqn_policy,double_targets
from simulator import Env,Config,SPLITS,load_data,rollout,summarize

def objective(trace):return (int((trace.excess>0).sum()),int(trace.excess.sum()),float(trace.cost.sum()))
def scenario(data,factor):
    # No validation/test rows are used to form a training scenario.
    d={k:v.copy() for k,v in data.items()}
    d['loads'][384:]=0;d['migrations'][385:]=0;d['live'][385:]=0
    scaled=data['loads'][:384].astype(float)*factor
    rounded=np.floor(scaled).astype(np.int64)
    totals=np.rint(data['loads'][:384,0].sum(1)*factor).astype(np.int64)
    for i,n in enumerate(range(2,9)):
        for t in range(384):
            remaining=int(totals[t]-rounded[t,i,:n].sum())
            order=np.argsort(-(scaled[t,i,:n]-rounded[t,i,:n]),kind='stable')
            rounded[t,i,order[:remaining]]+=1
    d['loads'][:384]=rounded
    assert np.all(d['loads'][:384].sum(2)==totals[:,None])
    return d

def reward(row,c,cost_weight):
    # Safety penalty remains fixed; independently optimize the server/migration tradeoff.
    operation=row['server_cost']+row['migration_cost']
    return -(cost_weight*operation+30*row['overload_cost']+10*int(row['excess']>0))/10

def train(data,c,initial,hp,seed,episodes,checkpoint=40,learning_rate=None):
    rng=np.random.default_rng(seed);net=MLP.load(initial);target=MLP.load(initial)
    factors=(.75,1.,1.25,1.5,2.) if hp['stress'] else (1.,)
    scenarios=[scenario(data,f) for f in factors]
    capacity=30000;x=np.zeros((capacity,10));nx=x.copy();a=np.zeros(capacity,int);r=np.zeros(capacity);done_buffer=r.copy()
    count=0;step=0;history=[]
    baseline=rollout(data,c,'validation',dqn_policy(net));best=objective(baseline);best_weights=[p.copy() for p in net.p]
    history.append(dict(episode=0,overloaded_slots=best[0],excess_requests=best[1],cost=best[2],loss=None))
    lr=hp['lr'] if learning_rate is None else learning_rate
    for ep in range(episodes):
        chosen=scenarios[int(rng.integers(len(scenarios)))];env=Env(chosen,c,0,384);s=env.reset();done=False;losses=[]
        epsilon=max(.02,.25*(1-ep/(episodes*.8)))
        while not done:
            action=int(rng.integers(3)) if rng.random()<epsilon else int(net.forward(s).argmax())
            ns,_,done,row=env.step(action-1);i=count%capacity
            x[i]=s;nx[i]=ns;a[i]=action;r[i]=reward(row,c,hp['cost_weight']);done_buffer[i]=done
            count+=1;step+=1;s=ns
            if count>=256 and step%4==0:
                idx=rng.integers(0,min(count,capacity),64)
                targets=double_targets(r[idx],done_buffer[idx],net.forward(nx[idx]),target.forward(nx[idx]),hp['gamma'])
                losses.append(net.update(x[idx],a[idx],targets,lr))
            if step%256==0:target.copy_from(net)
        if (ep+1)%checkpoint==0 or ep+1==episodes:
            val=rollout(data,c,'validation',dqn_policy(net));key=objective(val)
            history.append(dict(episode=ep+1,overloaded_slots=key[0],excess_requests=key[1],cost=key[2],loss=float(np.mean(losses))))
            if key<best:best=key;best_weights=[p.copy() for p in net.p]
    net.p=best_weights
    return net,history

def study(root,out,screen_episodes=200,final_episodes=600):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=load_data(root/'data/processed.npz');c=Config(**json.loads((root/'results/config.json').read_text()))
    s=json.loads((root/'results/safety/selection.json').read_text());initial=root/'results/safety'/(s['name']+'.npz')
    configs=[dict(cost_weight=w,gamma=g,stress=stress,lr=.0001) for w,g,stress in itertools.product((1.,2.,4.),(.9,.99),(False,True))]
    protocol=dict(configs=configs,screen_episodes=screen_episodes,final_episodes=final_episodes,screen_seed=7,final_seeds=[7,42,123],finalists=3,checkpoint=40,initial=str(initial.relative_to(root)),initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),selection='validation lexicographic overload count, excess, original cost; incumbent always included',synthetic_stress='TRAIN-ONLY demand multipliers 0.75,1,1.25,1.5,2; routing and live inventory unchanged. Explicit sensitivity scenarios, not new real observations.',test='Known test, repeated diagnostic evaluation. No test-based selection.',publication='Local only; no commit or push')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.time();screen=[]
    for trial,hp in enumerate(configs):
        net,h=train(data,c,initial,hp,7,screen_episodes)
        name=f'screen_{trial}';net.save(out/(name+'.npz'));pd.DataFrame(h).to_csv(out/(name+'_history.csv'),index=False)
        row=dict(trial=trial,**hp,**summarize(rollout(data,c,'validation',dqn_policy(net))));screen.append(row)
        pd.DataFrame(screen).to_csv(out/'screen.csv',index=False)
        print('Screen',trial+1,'/12',tuple(row[k] for k in ('overloaded_slots','excess_requests','cost')),flush=True)
    finalists=sorted(screen,key=lambda x:(x['overloaded_slots'],x['excess_requests'],x['cost']))[:3];rows=[]
    for finalist in finalists:
        hp={k:finalist[k] for k in ('cost_weight','gamma','stress','lr')}
        for seed in (7,42,123):
            name=f"trial_{finalist['trial']}_seed_{seed}"
            # Continue from best screened weights, fresh Adam and replay, smaller learning rate.
            net,h=train(data,c,out/f"screen_{finalist['trial']}.npz",hp,seed,final_episodes,learning_rate=.00005)
            net.save(out/(name+'.npz'));pd.DataFrame(h).to_csv(out/(name+'_history.csv'),index=False)
            row=dict(name=name,trial=finalist['trial'],seed=seed,**hp,**summarize(rollout(data,c,'validation',dqn_policy(net))));rows.append(row)
            pd.DataFrame(rows).to_csv(out/'finalists.csv',index=False)
            print('Final',name,tuple(row[k] for k in ('overloaded_slots','excess_requests','cost')),flush=True)
    df=pd.DataFrame(rows);means=df.groupby('trial')[['overloaded_slots','excess_requests','cost']].mean();best_trial=min(means.index,key=lambda i:tuple(means.loc[i]))
    candidates=[r for r in rows if r['trial']==best_trial];selected=min(candidates,key=lambda x:(x['overloaded_slots'],x['excess_requests'],x['cost']))
    selected['study_seconds']=time.time()-start
    (out/'selection.json').write_text(json.dumps(selected,indent=2))
    # Frozen selection above; no decisions from test metrics below.
    traces={'previous_safety':rollout(data,c,'test',dqn_policy(MLP.load(initial))),'cost_finetuned':rollout(data,c,'test',dqn_policy(MLP.load(out/(selected['name']+'.npz'))))}
    summary=pd.DataFrame({name:summarize(t) for name,t in traces.items()}).T;summary.to_csv(out/'comparison.csv')
    for name,trace in traces.items():trace.to_csv(out/('test_'+name+'.csv'),index=False)
    seed_rows=[]
    for r0 in candidates:
        trace=rollout(data,c,'test',dqn_policy(MLP.load(out/(r0['name']+'.npz'))));seed_rows.append(dict(seed=r0['seed'],**summarize(trace)))
    pd.DataFrame(seed_rows).to_csv(out/'test_seeds.csv',index=False)
    print('Selected',selected['name'],flush=True);print(summary.to_string(),flush=True)
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project-root',default=str(Path(__file__).resolve().parent));p.add_argument('--output',default='results/cost_finetuning');a=p.parse_args();study(a.project_root,a.output)
