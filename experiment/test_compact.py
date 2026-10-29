import unittest
import numpy as np
from data_pipeline import jch
from compact_data import compact,CompactEnv
from v2.environment import Env,Config
class CompactTests(unittest.TestCase):
    def test_every_step_equivalent_with_switches_and_queues(self):
        rng=np.random.default_rng(712);keys=rng.integers(0,2**63,200,dtype=np.int64)
        d={'mapping':np.array([[jch(int(k),n) for k in keys] for n in range(2,9)]),'work':rng.integers(0,10,(65,200))/4,'writes':rng.integers(0,2,(65,200)),'inventory':rng.integers(0,30,(66,200))}
        cd=compact(d);self.assertLess(cd['mapping'].shape[1],200)
        c=Config(100,500,1000);full=Env(d,c,1,65);small=CompactEnv(cd,c,1,65)
        np.testing.assert_allclose(full.observe(),small.observe(),rtol=1e-12,atol=1e-12)
        for action in [1]*8+[-1]*12+[1]*20+[0]*24:
            a,r,done,row=full.step(action);b,s,ended,other=small.step(action)
            np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-11)
            self.assertAlmostEqual(r,s,places=9)
            for k in row:self.assertAlmostEqual(row[k],other[k],places=8,msg=k)
    def test_group_peak_is_not_mistaken_for_hot_key(self):
        d={'mapping':np.zeros((7,20),dtype=int),'work':np.ones((4,20)),'writes':np.zeros((4,20)),'inventory':np.zeros((5,20))}
        e=CompactEnv(compact(d),Config(10,100,100),0,2)
        row=e.step(0)[3];self.assertEqual(row['hot_key_over_capacity'],0);self.assertEqual(row['arrival_over_capacity'],1)
if __name__=='__main__':unittest.main()
