import unittest,itertools,tempfile
from pathlib import Path
import numpy as np
from safe_finetuning import adjusted_reward,oracle_trace,fine_tune,metric
from simulator import Config,Env,SPLITS
from agents import MLP
class SafetyTests(unittest.TestCase):
    def data(self):return dict(loads=np.zeros((672,7,8),dtype=np.int64),migrations=np.zeros((673,7,7),dtype=np.int64),live=np.zeros(673,dtype=np.int64))
    def test_common_cost_unchanged(self):
        c=Config(10,100);row={'cost':3,'excess':0}
        self.assertEqual(adjusted_reward(row,c,30,10),-.3)
        row={'cost':5,'excess':1};before=row.copy()
        self.assertLess(adjusted_reward(row,c,30,10),adjusted_reward(row,c,3,0));self.assertEqual(row,before)
    def test_oracle_matches_exhaustive_search(self):
        d=self.data();d['loads'][:3]=np.arange(3*7*8).reshape(3,7,8)%17
        c=Config(10,100);SPLITS['tiny']=(0,3)
        try:
            actual=metric(oracle_trace(d,c,'tiny'));candidates=[]
            for actions in itertools.product((-1,0,1),repeat=3):
                e=Env(d,c,0,3);rows=[e.step(a)[3] for a in actions]
                import pandas as pd
                candidates.append(metric(pd.DataFrame(rows)))
            self.assertEqual(actual,min(candidates))
        finally:del SPLITS['tiny']
    def test_warm_start_and_test_isolation(self):
        # 30 episodes covers validation checkpoint; test changes cannot affect selected weights.
        d=self.data();c=Config(10,100);hp=dict(multiplier=3,event=0)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'net.npz';MLP(8).save(p)
            a,_=fine_tune(d,c,p,hp,episodes=30)
            d['loads'][480:]=999999;d['migrations'][481:]=999999
            b,_=fine_tune(d,c,p,hp,episodes=30)
            for x,y in zip(a.p,b.p):np.testing.assert_array_equal(x,y)
