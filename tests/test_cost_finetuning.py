import unittest,tempfile
from pathlib import Path
import numpy as np
from cost_finetuning import scenario,reward,train
from simulator import Config
from agents import MLP
class CostTests(unittest.TestCase):
 def data(self):
  d=dict(loads=np.zeros((672,7,8),dtype=np.int64),migrations=np.zeros((673,7,7),dtype=np.int64),live=np.zeros(673,dtype=np.int64))
  for i,n in enumerate(range(2,9)):d['loads'][:,i,0]=17;d['loads'][:,i,1]=8
  return d
 def test_stress_conservation_and_no_mutation(self):
  d=self.data();copy=d['loads'].copy();s=scenario(d,1.25)
  np.testing.assert_array_equal(d['loads'],copy)
  np.testing.assert_array_equal(s['loads'][:384].sum(2),np.full((384,7),31))
  self.assertFalse(s['loads'][384:].any())
 def test_scenario_ignores_holdout(self):
  d=self.data();a=scenario(d,2)
  d['loads'][384:]=999999;d['migrations'][385:]=999;d['live'][385:]=999
  b=scenario(d,2)
  for key in a:np.testing.assert_array_equal(a[key],b[key])
 def test_cost_priority_without_rewriting_common_metrics(self):
  row=dict(server_cost=3,migration_cost=1,overload_cost=2,excess=1,cost=6);copy=row.copy()
  self.assertLess(reward(row,Config(10,100),4),reward(row,Config(10,100),1));self.assertEqual(row,copy)
 def test_incumbent_is_available(self):
  d=self.data();hp=dict(cost_weight=2,gamma=.9,stress=False,lr=.0001)
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'net.npz';MLP(8).save(p)
   n,h=train(d,Config(10,100),p,hp,7,episodes=1,checkpoint=1)
   base=h[0];best=min(h,key=lambda r:(r['overloaded_slots'],r['excess_requests'],r['cost']))
   self.assertLessEqual((best['overloaded_slots'],best['excess_requests'],best['cost']),(base['overloaded_slots'],base['excess_requests'],base['cost']))
