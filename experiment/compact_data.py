"""Exact routing-signature aggregation for proportional fluid service.
Keys sharing their entire 2..8 JCH routing vector receive the same service
fraction and always migrate together. Their queues can therefore be summed.
The maximum ORIGINAL key work is preserved separately for the hot-key feature.
"""
from pathlib import Path
import numpy as np
from v2.prepare import load
from v2.environment import Env

def compact(d):
    signatures,inverse=np.unique(np.asarray(d['mapping']).T,axis=0,return_inverse=True)
    result={'mapping':signatures.T,'hot_work':np.asarray(d['work']).max(1)}
    for name in ('work','writes','inventory','counts'):
        if name not in d:continue
        rows=[]
        for row in d[name]:rows.append(np.bincount(inverse,weights=row,minlength=len(signatures)))
        result[name]=np.asarray(rows)
    return result

def save_compact(source,target):
    d=load(source);result=compact(d);target=Path(target);target.mkdir(exist_ok=True)
    for name,value in result.items():np.save(target/(name+'.npy'),value)
    print('Compacted',source,'keys',d['mapping'].shape[1],'groups',result['mapping'].shape[1],flush=True)

def load_compact(path):
    return {p.stem:np.load(p) for p in Path(path).glob('*.npy')}

class CompactEnv(Env):
    def reset(self):
        super().reset()
        self.last_hot=float(self.d['hot_work'][self.t-1]/self.c.capacity) if self.t else 0.
        return self.observe()
    def step(self,a):
        t=self.t;_,reward,done,row=super().step(a)
        self.last_hot=float(self.d['hot_work'][t]/self.c.capacity)
        row['hot_key_over_capacity']=int(self.d['hot_work'][t]>self.c.capacity)
        return self.observe(),reward,done,row

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('target');a=p.parse_args();save_compact(a.source,a.target)
