"""Causal request-capacity simulation; model units, no measured ClickHouse latency."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
SPLITS={'train':(0,384),'validation':(384,480),'test':(480,672)}
@dataclass(frozen=True)
class Config:
    capacity:float
    migration_scale:float
    shard_price:float=1.0
    overload_price:float=20.0
    migration_price:float=4.0

def load_data(path='data/processed.npz'):
    with np.load(path) as d:return {k:d[k] for k in d.files}

def calibrate(data):
    # Only train. n=4 maps to index 2. Physical capacity is a declared scenario assumption.
    return Config(capacity=max(1,float(np.ceil(np.quantile(data['loads'][:384,2,:4].max(1),.75)))),migration_scale=max(1,float(np.median(data['live'][:384]))))

class Env:
    def __init__(self,data,config,start,end,initial=2):
        self.data=data;self.c=config;self.start=start;self.end=end;self.initial=initial
        self.reset()
    def reset(self):
        self.t=self.start;self.n=self.initial
        # Past requests are available at deployment, assuming initial placement in prior slots.
        self.history=[]
        for t in range(max(0,self.start-4),self.start):
            v=self.data['loads'][t,self.n-2,:self.n]
            self.history.append((float(v.sum()),float(v.max())))
        return self.observe()
    def observe(self):
        c=self.c;hist=self.history
        total,maxload=hist[-1] if hist else (0.,0.)
        previous=hist[-2][0] if len(hist)>1 else total
        mean=sum(h[0] for h in hist[-4:])/min(4,len(hist)) if hist else 0.
        trend=(total-previous)/c.capacity
        angle=2*np.pi*(self.t%96)/96
        i=self.n-2;m=self.data['migrations'][min(self.t,len(self.data['migrations'])-1)]
        down=float(m[i,max(0,i-1)])/c.migration_scale
        up=float(m[i,min(6,i+1)])/c.migration_scale
        return np.array([(self.n-2)/6,total/(8*c.capacity),maxload/c.capacity,mean/(8*c.capacity),trend/8,np.sin(angle),np.cos(angle),self.data['live'][self.t]/c.migration_scale,down,up],dtype=np.float64)
    def step(self,action):
        if self.t>=self.end or action not in (-1,0,1):raise ValueError('Invalid step')
        new=min(8,max(2,self.n+action));c=self.c
        moved=int(self.data['migrations'][self.t,self.n-2,new-2])
        demand=self.data['loads'][self.t,new-2,:new]
        excess=int(np.maximum(demand-c.capacity,0).sum())
        cost_parts=(new*c.shard_price,c.overload_price*excess/c.capacity,c.migration_price*moved/c.migration_scale)
        row=dict(old_shards=self.n,slot=self.t,hour=self.t/4,shards=new,requests=int(demand.sum()),max_load=int(demand.max()),excess=excess,moved_bytes=moved,server_cost=cost_parts[0],overload_cost=cost_parts[1],migration_cost=cost_parts[2],cost=sum(cost_parts))
        self.history.append((float(demand.sum()),float(demand.max())));self.n=new;self.t+=1
        return self.observe(),-row['cost'],self.t>=self.end,row

def rollout(data,config,split,policy,initial=2):
    env=Env(data,config,*SPLITS[split],initial=initial);s=env.reset();done=False;rows=[]
    while not done:s,r,done,row=env.step(int(policy(s)));rows.append(row)
    return pd.DataFrame(rows)

def threshold(up=1.0,down=.55):
    return lambda s:1 if s[2]>up else (-1 if s[2]<down else 0)

def summarize(trace):
    return dict(cost=float(trace.cost.sum()),server_cost=float(trace.server_cost.sum()),overload_cost=float(trace.overload_cost.sum()),migration_cost=float(trace.migration_cost.sum()),overloaded_slots=int((trace.excess>0).sum()),excess_requests=int(trace.excess.sum()),moved_mb=float(trace.moved_bytes.sum()/1e6),mean_shards=float(trace.shards.mean()),changes=int((trace.shards!=trace.old_shards).sum()))

def baselines(data,c,split,up=1.,down=.55):
    traces={f'fixed_{n}':rollout(data,c,split,lambda s:0,initial=n) for n in range(2,9)}
    traces['threshold']=rollout(data,c,split,threshold(up,down))
    return traces
