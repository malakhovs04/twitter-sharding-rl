"""Fluid per-key queue, bandwidth-limited snapshot+dirty-log transfer, atomic cutover.
This is a declared synthetic scenario, not a calibrated production emulator.
"""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class Config:
    capacity:float
    bandwidth:float
    migration_scale:float
    migration_cpu_fraction:float=.20
    server_price:float=1.
    transfer_price:float=4.
    queue_price:float=20.
    terminal_queue_price:float=20.

def calibrate(data,start=96,end=384):
    # Target average pooled utilization 65% for four shards, not a peak percentile.
    cap=max(1.,float(data['work'][start:end].sum(1).mean())/(4*.65))
    scale=max(1.,float(np.median(data['inventory'][start:end].sum(1))))
    # Full live inventory takes 8 slots at this scenario bandwidth.
    return Config(capacity=cap,bandwidth=scale/8,migration_scale=scale)

class Env:
    def __init__(self,data,c,start,end,initial=4):
        self.d=data;self.c=c;self.start=start;self.end=end;self.initial=initial;self.reset()
    def reset(self):
        self.t=self.start;self.n=self.initial;self.target=self.n;self.remaining=0.;self.mask=None
        self.queue=np.zeros(self.d['work'].shape[1]);self.last_util=0.;self.last_hot=0.;self.last_total=0.;self.trend=0.;self.migration_age=0
        if self.t:
            last=self.d['work'][self.t-1];load=np.bincount(self.d['mapping'][self.n-2],weights=last,minlength=self.n)
            self.last_util=float(load.max()/self.c.capacity);self.last_hot=float(last.max()/self.c.capacity);self.last_total=float(last.sum()/self.c.capacity)
        return self.observe()
    def observe(self):
        # All features use state at the boundary or completed intervals; never work[t].
        return np.array([(self.n-2)/6,self.last_util,self.queue.sum()/self.c.capacity,self.remaining/self.c.migration_scale,
            (self.target-self.n),self.last_hot,self.last_total/8,self.trend/8,
            self.d['inventory'][self.t].sum()/self.c.migration_scale,self.migration_age/8],dtype=float)
    def step(self,a):
        if self.t>=self.end or a not in (-1,0,1):raise ValueError('invalid action/time')
        old=self.n;started=False;completed=False;busy=self.target!=self.n
        if not busy and a:
            target=int(np.clip(self.n+a,2,8))
            if target!=self.n:
                self.target=target;self.mask=self.d['mapping'][self.n-2]!=self.d['mapping'][target-2]
                self.remaining=float(self.d['inventory'][self.t,self.mask].sum());self.migration_age=0;started=True
        active=self.target!=self.n
        # Destination servers billed from launch, source servers retained until cutover.
        billed=max(self.n,self.target);snapshot=self.remaining;dirty=0.;transferred=0.
        if active:
            dirty=float(self.d['writes'][self.t,self.mask].sum());self.remaining+=dirty
            transferred=min(self.remaining,self.c.bandwidth);self.remaining-=transferred;self.migration_age+=1
        # Current interval served on source layout; migration consumes capacity on source servers.
        effective=self.c.capacity*(1-self.c.migration_cpu_fraction*min(1.,transferred/self.c.bandwidth))
        incoming=np.asarray(self.d['work'][self.t]);backlog_before=float(self.queue.sum())
        pending=self.queue+incoming;mapping=self.d['mapping'][self.n-2]
        per_shard=np.bincount(mapping,weights=pending,minlength=self.n)
        fraction=np.minimum(1.,effective/np.maximum(per_shard,1e-12))
        served=pending*fraction[mapping];self.queue=np.maximum(0.,pending-served)
        demand=np.bincount(mapping,weights=incoming,minlength=self.n)
        self.last_util=float(demand.max()/self.c.capacity);self.last_hot=float(incoming.max()/self.c.capacity)
        total=float(incoming.sum()/self.c.capacity);self.trend=total-self.last_total;self.last_total=total
        # Per-key pending work survives cutover exactly; routed on new layout next interval.
        if active and self.remaining<=1e-9:
            self.remaining=0.;self.n=self.target;self.mask=None;completed=True;self.migration_age=0
        server_cost=billed*self.c.server_price
        transfer_cost=transferred/self.c.migration_scale*self.c.transfer_price
        queue_cost=self.queue.sum()/self.c.capacity*self.c.queue_price
        self.t+=1;done=self.t>=self.end
        terminal_queue_cost=self.queue.sum()/self.c.capacity*self.c.terminal_queue_price if done else 0.
        # Residual transfer obligation is reported and priced at horizon; it is NOT claimed completed.
        terminal_transfer_cost=self.remaining/self.c.migration_scale*self.c.transfer_price if done else 0.
        cost=float(server_cost+transfer_cost+queue_cost+terminal_queue_cost+terminal_transfer_cost)
        row=dict(slot=self.t-1,old_shards=old,shards=self.n,target=self.target,billed_shards=billed,action=a,
            migration_started=int(started),migration_active=int(active),migration_completed=int(completed),
            transfer_before=snapshot,dirty_bytes=dirty,transferred_bytes=transferred,transfer_remaining=self.remaining,
            incoming_work=float(incoming.sum()),served_work=float(served.sum()),queue_before=backlog_before,queue_after=float(self.queue.sum()),
            queue_capacity_units=float(self.queue.sum()/self.c.capacity),queue_violation=int(self.queue.sum()>1e-7),
            hot_key_over_capacity=int(incoming.max()>self.c.capacity),arrival_over_capacity=int(demand.max()>effective),
            server_cost=server_cost,transfer_cost=transfer_cost,queue_cost=queue_cost,terminal_queue_cost=terminal_queue_cost,
            terminal_transfer_cost=terminal_transfer_cost,cost=cost)
        return self.observe(),-cost/10,done,row

def rollout(data,c,start,end,policy,initial=4):
    e=Env(data,c,start,end,initial);s=e.reset();rows=[];done=False
    while not done:s,_,done,row=e.step(int(policy(s)));rows.append(row)
    return rows

def summary(rows):
    import pandas as pd
    f=pd.DataFrame(rows)
    return dict(cost=float(f.cost.sum()),server_cost=float(f.server_cost.sum()),transfer_cost=float(f.transfer_cost.sum()),queue_cost=float(f.queue_cost.sum()),
        queue_slots=int(f.queue_violation.sum()),queue_work_area=float(f.queue_after.sum()),terminal_queue=float(f.queue_after.iloc[-1]),
        terminal_transfer=float(f.transfer_remaining.iloc[-1]),moved_mb=float(f.transferred_bytes.sum()/1e6),migration_slots=int(f.migration_active.sum()),
        migrations_started=int(f.migration_started.sum()),migrations_completed=int(f.migration_completed.sum()),hot_key_slots=int(f.hot_key_over_capacity.sum()))

# Exactly the same discretization is supplied to table and matched DQN.
EDGES=[np.array([.1,.25,.45,.6,.8,.95]),np.array([.5,.8,1,1.3,2]),np.array([.01,.25,1,4]),np.array([.001,.1,.5,1]),np.array([-.5,.5]),np.array([.5,1]),np.array([.3,.6,1]),np.array([-.1,.1]),np.array([.5,1,1.5]),np.array([.1,.5,1])]
def state_key(s):return tuple(int(np.searchsorted(e,x,side='right')) for x,e in zip(s,EDGES))
def encode(s,matched=True):
    if matched:return np.array([k/(len(e)+1) for k,e in zip(state_key(s),EDGES)],dtype=float)
    return np.clip(np.asarray(s),-5,10)
