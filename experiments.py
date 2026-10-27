"""Predeclared train/validation search; test is evaluated only in evaluate()."""
from pathlib import Path
from dataclasses import asdict
import itertools,json,time,hashlib,platform
import numpy as np
import pandas as pd
from simulator import *
from agents import *
ROOT=Path('results')
def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2))
def config():return Config(**json.loads((ROOT/'config.json').read_text()))
def log(message):
    print(message,flush=True)
    with (ROOT/'progress.log').open('a') as f:f.write(message+'\n')
def initialize():
    ROOT.mkdir(exist_ok=True);data=load_data();c=calibrate(data)
    write_json(ROOT/'config.json',asdict(c))
    protocol={'split_slots':SPLITS,'capacity_rule':'train 75% quantile of maximum per-shard requests at n=4','inventory_scale_rule':'train median live bytes','scenario_costs':asdict(c),'dqn_screen':{'hidden':[32,64],'lr':[.0003,.001],'gamma':[.90,.97,.99],'seed':7,'episodes':160,'checkpoint':40},'dqn_finalists':3,'dqn_final_seeds':[7,42,123],'dqn_final_episodes':400,'dqn_checkpoint':40,'replay_capacity':20000,'batch':64,'warmup':256,'update_every':4,'target_interval':256,'reward_scale':10,'q_alpha':[.05,.15],'q_gamma':[.90,.97],'q_episodes':600,'q_seeds':[7,42,123],'test_policy':'One evaluation after all selections; no retuning on test','python':platform.python_version(),'numpy':np.__version__,'data_sha256':hashlib.sha256(Path('data/processed.npz').read_bytes()).hexdigest()}
    write_json(ROOT/'protocol.json',protocol)
    # Test not touched for any decisions below.
    train_table=pd.DataFrame({f'n={n}':data['loads'][:384,n-2,:n].max(1) for n in range(2,9)})
    train_table.to_csv(ROOT/'train_loads.csv',index=False)
    log('Protocol fixed. Capacity='+str(c.capacity))
    return c

def run_baselines():
    data=load_data();c=config();rows=[]
    for up,down in itertools.product((.8,1.,1.2),(.4,.6)):
        trace=rollout(data,c,'validation',threshold(up,down));rows.append(dict(up=up,down=down,**summarize(trace)))
    search=pd.DataFrame(rows);search.to_csv(ROOT/'threshold_search.csv',index=False);chosen=min(rows,key=lambda r:r['cost'])
    fixed=baselines(data,c,'validation');fixed_rows={k:summarize(v) for k,v in fixed.items() if k.startswith('fixed')}
    best_fixed=min(fixed_rows,key=lambda k:fixed_rows[k]['cost'])
    write_json(ROOT/'baseline_selection.json',dict(up=chosen['up'],down=chosen['down'],fixed=best_fixed))
    train_traces=baselines(data,c,'train',chosen['up'],chosen['down'])
    pd.DataFrame({k:summarize(v) for k,v in train_traces.items()}).T.to_csv(ROOT/'baseline_train.csv')
    log('Baselines validated: '+best_fixed+'; threshold '+str((chosen['up'],chosen['down'])))
    return search

def save_q(path,q):write_json(path,[{'state':list(k),'q':v.tolist()} for k,v in q.items()])
def load_q(path):return {tuple(r['state']):np.array(r['q']) for r in json.loads(Path(path).read_text())}

def run_q_study(retrain=False):
    folder=ROOT/'q';folder.mkdir(exist_ok=True)
    if (folder/'selection.json').exists() and not retrain:return pd.read_csv(folder/'search.csv')
    data=load_data();c=config();rows=[]
    for alpha,gamma in itertools.product((.05,.15),(.90,.97)):
        for seed in (7,42,123):
            name=f'a{alpha}_g{gamma}_s{seed}'
            q,h=train_q(data,c,seed=seed,episodes=600,alpha=alpha,gamma=gamma)
            save_q(folder/(name+'.json'),q);pd.DataFrame(h).to_csv(folder/(name+'_history.csv'),index=False)
            val=float(rollout(data,c,'validation',q_policy(q)).cost.sum())
            rows.append(dict(name=name,alpha=alpha,gamma=gamma,seed=seed,validation_cost=val))
        log(f'Q-learning alpha={alpha}, gamma={gamma} complete')
    search=pd.DataFrame(rows);search.to_csv(folder/'search.csv',index=False)
    means=search.groupby(['alpha','gamma']).validation_cost.mean();hp=means.idxmin()
    selected=search[(search.alpha==hp[0])&(search.gamma==hp[1])].sort_values('validation_cost',kind='stable').iloc[0].to_dict()
    write_json(folder/'selection.json',selected);log('Q selected '+selected['name'])
    return search

def run_dqn_study(retrain=False):
    folder=ROOT/'dqn';folder.mkdir(exist_ok=True)
    if (folder/'selection.json').exists() and not retrain:return pd.read_csv(folder/'finalists.csv')
    data=load_data();c=config();rows=[];started=time.time()
    for i,(hidden,lr,gamma) in enumerate(itertools.product((32,64),(.0003,.001),(.90,.97,.99))):
        hp=dict(hidden=hidden,lr=lr,gamma=gamma,target_interval=256)
        net,h=train_dqn(data,c,hp,seed=7,episodes=160,checkpoint=40)
        name=f'screen_{i}';net.save(folder/(name+'.npz'));pd.DataFrame(h).to_csv(folder/(name+'_history.csv'),index=False)
        val=float(rollout(data,c,'validation',dqn_policy(net)).cost.sum())
        rows.append(dict(trial=i,**hp,validation_cost=val))
        pd.DataFrame(rows).to_csv(folder/'screen.csv',index=False)
        log(f'DQN screen {i+1}/12 val={val:.2f}')
    finalists=sorted(rows,key=lambda r:r['validation_cost'])[:3];final_rows=[]
    for row in finalists:
        hp={k:row[k] for k in ('hidden','lr','gamma','target_interval')}
        for seed in (7,42,123):
            name=f"trial_{row['trial']}_seed_{seed}"
            net,h=train_dqn(data,c,hp,seed=seed,episodes=400,checkpoint=40)
            net.save(folder/(name+'.npz'));pd.DataFrame(h).to_csv(folder/(name+'_history.csv'),index=False)
            val=float(rollout(data,c,'validation',dqn_policy(net)).cost.sum())
            final_rows.append(dict(name=name,trial=row['trial'],seed=seed,**hp,validation_cost=val,best_episode=min(h,key=lambda x:x['validation_cost'])['episode']))
            pd.DataFrame(final_rows).to_csv(folder/'finalists.csv',index=False)
            log(f'DQN final {name} val={val:.2f}')
    table=pd.DataFrame(final_rows);mean=table.groupby('trial').validation_cost.mean();best_trial=int(mean.idxmin())
    chosen=table[table.trial==best_trial].sort_values('validation_cost',kind='stable').iloc[0].to_dict()
    chosen['selection_rule']='Minimum mean validation cost across 3 seeds, then best validation seed; best checkpoint by validation'
    chosen['study_seconds']=time.time()-started
    write_json(folder/'selection.json',chosen);log('DQN selected '+chosen['name'])
    return table

def oracle(data,c,split):
    # Offline perfect-future dynamic programming lower bound; never a deployable baseline.
    start,end=SPLITS[split];values=np.zeros((end-start+1,7));acts=np.zeros((end-start,7),dtype=int)
    for t in range(end-1,start-1,-1):
        for i in range(7):
            options=[]
            for a in (-1,0,1):
                j=min(6,max(0,i+a));n=j+2;v=data['loads'][t,j,:n]
                cost=n*c.shard_price+c.overload_price*np.maximum(v-c.capacity,0).sum()/c.capacity+c.migration_price*data['migrations'][t,i,j]/c.migration_scale
                options.append(cost+values[t-start+1,j])
            best=int(np.argmin(options));values[t-start,i]=options[best];acts[t-start,i]=best-1
    return float(values[0,0])

def evaluate():
    data=load_data();c=config();base=json.loads((ROOT/'baseline_selection.json').read_text());qs=json.loads((ROOT/'q/selection.json').read_text());ds=json.loads((ROOT/'dqn/selection.json').read_text())
    # Freeze exact selected artifacts before first test computation.
    frozen={'baseline':base,'q':qs,'dqn':ds,'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()}
    write_json(ROOT/'frozen_selection.json',frozen)
    traces=baselines(data,c,'test',base['up'],base['down'])
    traces['q_learning']=rollout(data,c,'test',q_policy(load_q(ROOT/'q'/(qs['name']+'.json'))))
    traces['dqn']=rollout(data,c,'test',dqn_policy(MLP.load(ROOT/'dqn'/(ds['name']+'.npz'))))
    summary=pd.DataFrame({k:summarize(v) for k,v in traces.items()}).T
    summary['selected_on_validation']=[k in (base['fixed'],'threshold','q_learning','dqn') for k in summary.index]
    summary.to_csv(ROOT/'comparison.csv')
    for name,trace in traces.items():trace.to_csv(ROOT/f'test_{name}.csv',index=False)
    seedrows=[]
    # All 3 seeds for chosen hp, no post-test reselection.
    finals=pd.read_csv(ROOT/'dqn/finalists.csv')
    for row in finals[finals.trial==ds['trial']].to_dict('records'):
        v=rollout(data,c,'test',dqn_policy(MLP.load(ROOT/'dqn'/(row['name']+'.npz'))))
        seedrows.append(dict(seed=row['seed'],**summarize(v)))
    pd.DataFrame(seedrows).to_csv(ROOT/'dqn/test_seeds.csv',index=False)
    write_json(ROOT/'oracle.json',{'cost':oracle(data,c,'test'),'description':'Perfect future offline lower bound, not an online strategy'})
    log('Final test complete; no retuning.');return summary

if __name__=='__main__':
    initialize();run_baselines();run_q_study();run_dqn_study();print(evaluate().to_string())
