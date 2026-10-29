"""Exploratory audit of an already viewed week; does not train or overwrite models."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import Env, SPLITS, load_data, calibrate, rollout, threshold, summarize
from agents import MLP, dqn_policy

OUT=Path('results/methodology_audit')

def select_threshold(data, config, objective):
    rows=[]
    # Filter 23*17 pairs by down < up; identical budget for both objectives.
    for up in np.linspace(.6,1.7,23):
        for down in np.linspace(.1,.9,17):
            if down>=up: continue
            row=summarize(rollout(data,config,'validation',threshold(up,down)))
            rows.append(dict(up=float(up),down=float(down),**row))
    frame=pd.DataFrame(rows)
    order=['cost','overloaded_slots','excess_requests','up','down'] if objective=='cost' else ['overloaded_slots','excess_requests','cost','up','down']
    best=frame.sort_values(order,kind='stable').iloc[0]
    return dict(up=float(best.up),down=float(best.down)),frame

def target_policy(n):
    return lambda s:int(np.sign(n-(int(round(s[0]*6))+2)))

def reprice(trace, config, overload, migration):
    return float(trace.server_cost.sum()+overload*trace.excess.sum()/config.capacity+migration*trace.moved_bytes.sum()/config.migration_scale)

def decision_trace(data,c,split,policy):
    env=Env(data,c,*SPLITS[split]);rows=[];s=env.reset();done=False
    while not done:
        a=int(policy(s));before=dict(previous_utilization=float(s[2]),previous_trend=float(s[4]),requested_action=a)
        s,_,done,row=env.step(a);rows.append(dict(**before,**row))
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=load_data();c=calibrate(data)
    original=json.loads(Path('results/dqn/selection.json').read_text())
    policies={'original_dqn':dqn_policy(MLP.load('results/dqn/'+original['name']+'.npz')),
              'safety_dqn_exploratory':dqn_policy(MLP.load('results/safety/m30_e10_s42.npz'))}
    old_threshold=json.loads(Path('results/baseline_selection.json').read_text())
    policies['original_threshold']=threshold(old_threshold['up'],old_threshold['down'])
    selections={}
    for objective in ('cost','safety'):
        chosen,grid=select_threshold(data,c,objective)
        grid.to_csv(OUT/f'threshold_{objective}_validation.csv',index=False)
        selections[objective]=dict(**chosen,candidates=len(grid))
        policies['threshold_'+objective]=threshold(**chosen)
    # Save choices before diagnostic replay of the already-seen test.
    (OUT/'selection.json').write_text(json.dumps(selections,indent=2))
    policies.update({f'ramp_target_{n}':target_policy(n) for n in range(2,9)})
    summaries=[];traces={}
    for name,policy in policies.items():
        trace=decision_trace(data,c,'test',policy);trace.to_csv(OUT/f'test_{name}.csv',index=False)
        traces[name]=trace;summaries.append(dict(strategy=name,**summarize(trace)))
    pd.DataFrame(summaries).to_csv(OUT/'comparison_common_start.csv',index=False)
    sensitivity=[]
    for name,trace in traces.items():
        for overload in (0,5,20,40,80,160):
            for migration in (0,1,4,16):
                sensitivity.append(dict(strategy=name,overload_price=overload,migration_price=migration,cost=reprice(trace,c,overload,migration)))
    pd.DataFrame(sensitivity).to_csv(OUT/'fixed_policy_repricing.csv',index=False)
    capacity=[]
    for factor in (.5,.75,1,1.25,1.5,2):
        scenario=replace(c,capacity=c.capacity*factor)
        for name,policy in policies.items():
            capacity.append(dict(strategy=name,capacity_factor=factor,**summarize(rollout(data,scenario,'test',policy))))
    pd.DataFrame(capacity).to_csv(OUT/'capacity_replay.csv',index=False)
    diagnostics=[]
    for day in range(7):
        lo=96*day;hi=lo+96
        peaks=np.array([data['loads'][lo:hi,i,:n].max(1) for i,n in enumerate(range(2,9))]).T
        diagnostics.append(dict(day=day+1,split='train' if day<4 else 'validation' if day==4 else 'seen_test',
            overload_slots_even_with_best_n=int((peaks.min(1)>c.capacity).sum()),
            mean_live_mb=float(data['live'][lo:hi].mean()/1e6),
            fixed4_overload_slots=int((peaks[:,2]>c.capacity).sum()),
            requests=int(data['loads'][lo:hi,0].sum())))
    pd.DataFrame(diagnostics).to_csv(OUT/'daily_diagnostics.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for row,name in enumerate(('original_dqn','safety_dqn_exploratory')):
        tr=traces[name]
        for action,label in ((-1,'remove'),(0,'hold'),(1,'add')):
            subset=tr[tr.requested_action==action]
            axes[row,0].scatter(subset.previous_utilization,subset.old_shards,label=label,alpha=.65,s=18)
        axes[row,0].set(xlabel='Observed previous utilization',ylabel='Shards before action',title=name)
        axes[row,0].legend();axes[row,1].step(tr.hour,tr.shards,where='post')
        axes[row,1].set(xlabel='Hour of week (already seen test)',ylabel='Shards')
    fig.suptitle('Observed decisions: associations, not a causal explanation or a universal rule')
    fig.tight_layout();fig.savefig(OUT/'policy_decisions.png',dpi=150);plt.close(fig)
    (OUT/'protocol.json').write_text(json.dumps(dict(status='EXPLORATORY: entire week previously viewed',initial_shards=2,action_limit=1,selection='validation only within already used week',repricing='Frozen action traces: no retraining; capacity sensitivity replays policies with changed observations',capacity=c.capacity,migration_scale=c.migration_scale,independent_generalization=False),indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(diagnostics).to_string(index=False))

if __name__=='__main__':main()
