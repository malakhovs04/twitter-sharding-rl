"""Tabular Q-learning and Double DQN, NumPy MLP with Adam and replay."""
from collections import defaultdict
import numpy as np
from simulator import Env,rollout,SPLITS

def state_key(s):
    n=int(round(s[0]*6))+2
    util=int(min(11,max(0,s[2]*4)))
    trend=int(s[4]>=-.025)+int(s[4]>=.025)
    # Known time-of-day only, never next-hour traffic.
    phase=int(((np.arctan2(s[5],s[6])%(2*np.pi))/(2*np.pi))*8)%8
    move=int(s[9]>=.1)+int(s[9]>=.4)
    return (n,util,trend,phase,move)

def q_policy(table):return lambda s:int(table.get(state_key(s),np.zeros(3)).argmax())-1

def train_q(data,c,seed=7,episodes=500,alpha=.1,gamma=.95):
    rng=np.random.default_rng(seed);q=defaultdict(lambda:np.zeros(3));history=[];best=float('inf');best_q=None
    for ep in range(episodes):
        env=Env(data,c,*SPLITS['train']);s=env.reset();done=False;cost=0
        eps=max(.05,1-ep/(episodes*.8))
        while not done:
            k=state_key(s);a=int(rng.integers(3)) if rng.random()<eps else int(q[k].argmax())
            ns,r,done,_=env.step(a-1);target=r+(0 if done else gamma*q[state_key(ns)].max())
            q[k][a]+=alpha*(target-q[k][a]);s=ns;cost-=r
        if (ep+1)%50==0:
            v=rollout(data,c,'validation',q_policy(q)).cost.sum();history.append(dict(episode=ep+1,validation_cost=float(v),exploration_cost=cost))
            if v<best:best=v;best_q={k:value.copy() for k,value in q.items()}
    return best_q if best_q is not None else dict(q),history

class MLP:
    def __init__(self,hidden=64,seed=7):
        rng=np.random.default_rng(seed);dims=[10,hidden,hidden,3];self.p=[]
        for a,b in zip(dims[:-1],dims[1:]):self.p.extend([rng.normal(0,np.sqrt(2/a),(a,b)),np.zeros(b)])
        self.m=[np.zeros_like(p) for p in self.p];self.v=[p.copy() for p in self.m];self.steps=0
    def forward(self,x):
        z=np.asarray(x)
        for i in range(0,len(self.p),2):
            z=np.einsum('...i,ij->...j',z,self.p[i])+self.p[i+1]
            if i<len(self.p)-2:z=np.maximum(0,z)
        return z
    def gradients(self,x,a,target):
        activations=[x];zs=[];z=x
        for i in range(0,len(self.p),2):
            z=np.einsum('...i,ij->...j',z,self.p[i])+self.p[i+1];zs.append(z)
            if i<len(self.p)-2:z=np.maximum(0,z)
            activations.append(z)
        error=z[np.arange(len(x)),a]-target
        loss=float(np.mean(np.where(abs(error)<=1,.5*error**2,abs(error)-.5)))
        delta=np.zeros_like(z);delta[np.arange(len(x)),a]=np.clip(error,-1,1)/len(x)
        grads=[None]*len(self.p)
        for layer in range(len(zs)-1,-1,-1):
            grads[2*layer]=np.einsum('bi,bj->ij',activations[layer],delta);grads[2*layer+1]=delta.sum(0)
            if layer:delta=np.einsum('ij,kj->ik',delta,self.p[2*layer])*(zs[layer-1]>0)
        return loss,grads
    def update(self,x,a,y,lr):
        loss,grads=self.gradients(x,a,y);self.steps+=1
        norm=np.sqrt(sum(float(np.sum(g*g)) for g in grads));scale=min(1,10/(norm+1e-12))
        for p,m,v,g in zip(self.p,self.m,self.v,grads):
            g=g*scale;m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
            p-=lr*(m/(1-.9**self.steps))/(np.sqrt(v/(1-.999**self.steps))+1e-8)
        if not all(np.isfinite(p).all() for p in self.p):raise FloatingPointError('Nonfinite weights')
        return loss
    def copy_from(self,net):self.p=[p.copy() for p in net.p]
    def save(self,path):np.savez_compressed(path,**{f'p{i}':p for i,p in enumerate(self.p)})
    @classmethod
    def load(cls,path):
        with np.load(path) as f:
            net=cls(hidden=f['p0'].shape[1]);net.p=[f[f'p{i}'].copy() for i in range(6)]
        return net

def dqn_policy(net):return lambda s:int(net.forward(s).argmax())-1

def double_targets(r,done,online_q,target_q,gamma):
    return r+gamma*(1-done)*target_q[np.arange(len(r)),online_q.argmax(1)]

def train_dqn(data,c,hp,seed=7,episodes=160,checkpoint=40):
    rng=np.random.default_rng(seed);net=MLP(hp['hidden'],seed);target=MLP(hp['hidden'],seed);target.copy_from(net)
    capacity=20000;xs=np.zeros((capacity,10));ns=xs.copy();actions=np.zeros(capacity,dtype=int);rs=np.zeros(capacity);ds=rs.copy()
    count=0;step=0;history=[];best=float('inf');best_p=None
    for ep in range(episodes):
        env=Env(data,c,*SPLITS['train']);s=env.reset();done=False;cost=0;losses=[]
        epsilon=max(.03,1-ep/(episodes*.8))
        while not done:
            a=int(rng.integers(3)) if rng.random()<epsilon else int(net.forward(s).argmax())
            nxt,r,done,_=env.step(a-1);i=count%capacity
            xs[i]=s;ns[i]=nxt;actions[i]=a;rs[i]=r/10;ds[i]=done;count+=1;step+=1;s=nxt;cost-=r
            if count>=256 and step%4==0:
                idx=rng.integers(0,min(count,capacity),size=64)
                y=double_targets(rs[idx],ds[idx],net.forward(ns[idx]),target.forward(ns[idx]),hp['gamma'])
                losses.append(net.update(xs[idx],actions[idx],y,hp['lr']))
            if step%hp['target_interval']==0:target.copy_from(net)
        if (ep+1)%checkpoint==0 or ep==episodes-1:
            val=float(rollout(data,c,'validation',dqn_policy(net)).cost.sum())
            history.append(dict(episode=ep+1,validation_cost=val,exploration_cost=cost,loss=float(np.mean(losses)),epsilon=epsilon))
            if val<best:best=val;best_p=[p.copy() for p in net.p]
    net.p=best_p
    return net,history
