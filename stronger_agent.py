"""Expanded causal network, train-only privileged teacher, then Double DQN refinement."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import pandas as pd
from simulator import Env,Config,load_data,SPLITS,rollout,summarize
from agents import MLP,double_targets
from cost_finetuning import scenario
INPUTS=52

def rank(trace):return (int((trace.excess>0).sum()),int(trace.excess.sum()),float(trace.cost.sum()))

class RichEnv(Env):
    def observe(self):
        old=super().observe();past=[]
        for lag in (1,2,4,8):
            past.extend(self.data['loads'][self.t-lag].max(1)/self.c.capacity if self.t>=lag else np.zeros(7))
        onehot=np.eye(7)[self.n-2]
        migrations=self.data['migrations'][self.t,self.n-2]/self.c.migration_scale
        return np.concatenate([old,np.array(past),onehot,migrations])

def rich_rollout(data,c,split,net,initial=2):
    env=RichEnv(data,c,*SPLITS[split],initial=initial);s=env.reset();done=False;rows=[]
    while not done:s,r,done,row=env.step(int(net.forward(s).argmax())-1);rows.append(row)
    return pd.DataFrame(rows)

class Network:
    def __init__(self,width=128,seed=7):
        rng=np.random.default_rng(seed);self.p=[]
        for a,b in ((INPUTS,width),(width,width),(width,3)):
            self.p.extend([rng.normal(0,np.sqrt(2/a),(a,b)),np.zeros(b)])
        self.reset_optimizer()
    def reset_optimizer(self):
        self.m=[np.zeros_like(p) for p in self.p];self.v=[p.copy() for p in self.m];self.steps=0
    @classmethod
    def expand(cls,old,width=128,seed=7):
        net=cls(width,seed);h=old.p[0].shape[1]
        if width<h:raise ValueError('Cannot shrink warm-start model')
        # Preserve the old mapping exactly; new channels start as a disconnected branch.
        net.p[0][:,:h]=0;net.p[0][:10,:h]=old.p[0];net.p[1][:h]=old.p[1]
        net.p[2][:h,:h]=old.p[2];net.p[2][h:,:h]=0;net.p[2][:h,h:]=0
        net.p[3][:h]=old.p[3];net.p[4][:h]=old.p[4];net.p[4][h:]=0;net.p[5]=old.p[5].copy()
        return net
    def forward(self,x,cache=False):
        x=np.asarray(x);z1=np.einsum('...i,ij->...j',x,self.p[0])+self.p[1];h1=np.maximum(0,z1)
        z2=np.einsum('...i,ij->...j',h1,self.p[2])+self.p[3];h2=np.maximum(0,z2)
        q=np.einsum('...i,ij->...j',h2,self.p[4])+self.p[5]
        return (q,(x,z1,h1,z2,h2)) if cache else q
    def backward(self,cache,dq):
        x,z1,h1,z2,h2=cache
        d2=np.einsum('ij,kj->ik',dq,self.p[4])*(z2>0)
        d1=np.einsum('ij,kj->ik',d2,self.p[2])*(z1>0)
        return [np.einsum('bi,bj->ij',x,d1),d1.sum(0),np.einsum('bi,bj->ij',h1,d2),d2.sum(0),np.einsum('bi,bj->ij',h2,dq),dq.sum(0)]
    def classification_gradient(self,x,y,weights):
        q,cache=self.forward(x,cache=True);shift=q-q.max(1,keepdims=True);e=np.exp(shift);prob=e/e.sum(1,keepdims=True)
        sw=weights[y];loss=-np.sum(sw*np.log(np.maximum(prob[np.arange(len(x)),y],1e-12)))/sw.sum()
        dq=prob.copy();dq[np.arange(len(x)),y]-=1;dq*=sw[:,None]/sw.sum()
        return float(loss),self.backward(cache,dq)
    def q_gradient(self,x,a,y):
        q,cache=self.forward(x,cache=True);error=q[np.arange(len(x)),a]-y
        loss=np.mean(np.where(abs(error)<=1,.5*error**2,abs(error)-.5));dq=np.zeros_like(q);dq[np.arange(len(x)),a]=np.clip(error,-1,1)/len(x)
        return float(loss),self.backward(cache,dq)
    def update(self,grads,lr):
        self.steps+=1;norm=np.sqrt(sum(float(np.sum(g*g)) for g in grads));scale=min(1,10/(norm+1e-12))
        for p,m,v,g in zip(self.p,self.m,self.v,grads):
            g=g*scale;m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
            p-=lr*(m/(1-.9**self.steps))/(np.sqrt(v/(1-.999**self.steps))+1e-8)
        if not all(np.isfinite(p).all() for p in self.p):raise FloatingPointError('Nonfinite network')
    def copy(self):
        n=Network(self.p[0].shape[1]);n.p=[p.copy() for p in self.p];return n
    def save(self,path):np.savez_compressed(path,**{f'p{i}':p for i,p in enumerate(self.p)})
    @classmethod
    def load(cls,path):
        with np.load(path) as d:
            n=cls(d['p0'].shape[1]);n.p=[d[f'p{i}'].copy() for i in range(6)]
        return n

def teacher_actions(data,c,end=384):
    """Labels may use future within TRAIN only; none of these future values enter actor input."""
    values=[[(0,0,0.) for _ in range(7)] for _ in range(end+1)];actions=np.zeros((end,7),int)
    for t in range(end-1,-1,-1):
        for i in range(7):
            candidates=[]
            for a in (-1,0,1):
                j=min(6,max(0,i+a));n=j+2;excess=int(np.maximum(data['loads'][t,j,:n]-c.capacity,0).sum())
                cost=n*c.shard_price+c.overload_price*excess/c.capacity+c.migration_price*data['migrations'][t,i,j]/c.migration_scale
                later=values[t+1][j];candidates.append((int(excess>0)+later[0],excess+later[1],cost+later[2]))
            best=min(range(3),key=lambda k:candidates[k]);values[t][i]=candidates[best];actions[t,i]=best
    return actions

def build_teacher_dataset(data,c,factors=(.5,.75,1.,1.25,1.5,2.,2.5)):
    xs=[];ys=[]
    for factor in factors:
        d=scenario(data,factor);labels=teacher_actions(d,c)
        for t in range(384):
            for n in range(2,9):
                env=RichEnv(d,c,t,t+1,initial=n);xs.append(env.observe());ys.append(labels[t,n-2])
    return np.array(xs),np.array(ys,dtype=int)

def imitate(data,c,initial,x,y,width,lr,seed=7,epochs=80):
    rng=np.random.default_rng(seed);net=Network.expand(MLP.load(initial),width,seed)
    initial_metric=rank(rich_rollout(data,c,'validation',net));best=initial_metric;best_net=net.copy();history=[dict(epoch=0,loss=None,overloaded_slots=best[0],excess_requests=best[1],cost=best[2])]
    counts=np.bincount(y,minlength=3);weights=np.minimum(3,len(y)/(3*np.maximum(counts,1)))
    for epoch in range(epochs):
        order=rng.permutation(len(x));losses=[]
        for start in range(0,len(x),256):
            idx=order[start:start+256];loss,g=net.classification_gradient(x[idx],y[idx],weights);net.update(g,lr);losses.append(loss)
        if (epoch+1)%5==0:
            trace=rich_rollout(data,c,'validation',net);key=rank(trace)
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),overloaded_slots=key[0],excess_requests=key[1],cost=key[2]))
            if key<best:best=key;best_net=net.copy()
    return best_net,history

def refine(data,c,initial,seed=7,episodes=200):
    rng=np.random.default_rng(seed);net=Network.load(initial);target=net.copy();scenarios=[scenario(data,f) for f in (.75,1.,1.25,1.5,2.)]
    capacity=20000;x=np.zeros((capacity,INPUTS));nx=x.copy();aa=np.zeros(capacity,int);rr=np.zeros(capacity);dd=rr.copy();count=0;step=0
    best=rank(rich_rollout(data,c,'validation',net));best_net=net.copy();history=[dict(episode=0,overloaded_slots=best[0],excess_requests=best[1],cost=best[2])]
    for ep in range(episodes):
        env=RichEnv(scenarios[int(rng.integers(len(scenarios)))],c,0,384);s=env.reset();done=False
        while not done:
            epsilon=max(.02,.2*(1-ep/(episodes*.8)));a=int(rng.integers(3)) if rng.random()<epsilon else int(net.forward(s).argmax())
            ns,r,done,row=env.step(a-1);i=count%capacity;x[i]=s;nx[i]=ns;aa[i]=a
            rr[i]=-(row['server_cost']+row['migration_cost']+30*row['overload_cost']+10*(row['excess']>0))/10;dd[i]=done;count+=1;step+=1;s=ns
            if count>=256 and step%4==0:
                idx=rng.integers(0,min(count,capacity),64);y=double_targets(rr[idx],dd[idx],net.forward(nx[idx]),target.forward(nx[idx]),.95)
                loss,g=net.q_gradient(x[idx],aa[idx],y);net.update(g,.00005)
            if step%256==0:target=net.copy()
        if (ep+1)%20==0:
            key=rank(rich_rollout(data,c,'validation',net));history.append(dict(episode=ep+1,overloaded_slots=key[0],excess_requests=key[1],cost=key[2]))
            if key<best:best=key;best_net=net.copy()
    return best_net,history

def study(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.time()
    data=load_data(root/'data/processed.npz');c=Config(**json.loads((root/'results/config.json').read_text()))
    s=json.loads((root/'results/safety/selection.json').read_text());initial=root/'results/safety'/(s['name']+'.npz')
    protocol=dict(inputs=52,widths=[96,128],learning_rates=[.0003,.001],imitation_epochs=80,teacher_factors=[.5,.75,1,1.25,1.5,2,2.5],teacher='Perfect-future planner within TRAIN only, lexicographic overload count, excess, cost. Privileged labels, not actor observations.',refine_finalists=2,refine_seeds=[7,42,123],refine_episodes=200,checkpoint='5 imitation epochs / 20 RL episodes',selection='Mean validation lexicographic risk/excess/cost across 3 seeds, then validation seed; incumbent retained',initial=str(initial.relative_to(root)),initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),comparison_note='More past routing information than old DQN; improvements cannot be attributed to network width alone.',test='Known test, repeated diagnostic assessment',publication='Local only, no commit or push')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    x,y=build_teacher_dataset(data,c);np.savez_compressed(out/'teacher_train.npz',x=x,y=y)
    print('Train-only teacher dataset',x.shape,'label counts',np.bincount(y).tolist(),flush=True)
    screens=[]
    for trial,(width,lr) in enumerate([(96,.0003),(96,.001),(128,.0003),(128,.001)]):
        net,h=imitate(data,c,initial,x,y,width,lr);name=f'screen_{trial}';net.save(out/(name+'.npz'));pd.DataFrame(h).to_csv(out/(name+'_history.csv'),index=False)
        row=dict(trial=trial,width=width,lr=lr,**summarize(rich_rollout(data,c,'validation',net)));screens.append(row);pd.DataFrame(screens).to_csv(out/'screen.csv',index=False)
        print('Imitation',trial+1,'/4',tuple(row[k] for k in ('overloaded_slots','excess_requests','cost')),flush=True)
    finalists=sorted(screens,key=lambda r:(r['overloaded_slots'],r['excess_requests'],r['cost']))[:2];rows=[]
    for screen in finalists:
        for seed in (7,42,123):
            name=f"trial_{screen['trial']}_seed_{seed}";net,h=refine(data,c,out/f"screen_{screen['trial']}.npz",seed=seed)
            net.save(out/(name+'.npz'));pd.DataFrame(h).to_csv(out/(name+'_history.csv'),index=False)
            row=dict(name=name,trial=screen['trial'],seed=seed,**summarize(rich_rollout(data,c,'validation',net)));rows.append(row);pd.DataFrame(rows).to_csv(out/'finalists.csv',index=False)
            print('RL refinement',name,tuple(row[k] for k in ('overloaded_slots','excess_requests','cost')),flush=True)
    frame=pd.DataFrame(rows);means=frame.groupby('trial')[['overloaded_slots','excess_requests','cost']].mean();best=min(means.index,key=lambda i:tuple(means.loc[i]));candidates=[r for r in rows if r['trial']==best]
    selected=min(candidates,key=lambda r:(r['overloaded_slots'],r['excess_requests'],r['cost']));selected['study_seconds']=time.time()-start;(out/'selection.json').write_text(json.dumps(selected,indent=2))
    model=Network.load(out/(selected['name']+'.npz'))
    oldnet=MLP.load(initial);traces={'previous_safety':rollout(data,c,'test',lambda s:int(oldnet.forward(s).argmax())-1),'stronger_agent':rich_rollout(data,c,'test',model)}
    summary=pd.DataFrame({name:summarize(t) for name,t in traces.items()}).T;summary.to_csv(out/'comparison.csv')
    for name,trace in traces.items():trace.to_csv(out/(name+'.csv'),index=False)
    seedrows=[]
    for candidate in candidates:
        trace=rich_rollout(data,c,'test',Network.load(out/(candidate['name']+'.npz')));seedrows.append(dict(seed=candidate['seed'],**summarize(trace)))
    pd.DataFrame(seedrows).to_csv(out/'test_seeds.csv',index=False)
    print('Selected',selected['name']);print(summary.to_string(),flush=True);return summary
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',default=str(Path(__file__).resolve().parent));parser.add_argument('--output',default='results/stronger_agent');args=parser.parse_args();study(args.project_root,args.output)
