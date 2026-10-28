import unittest
import numpy as np
from simulator import Config,SPLITS
from operational_start import initial_shards,evaluate_policy
class OperationalTests(unittest.TestCase):
 def data(self):return dict(loads=np.zeros((672,7,8),dtype=np.int64),migrations=np.zeros((673,7,7),dtype=np.int64),live=np.zeros(673,dtype=np.int64))
 def test_decision_ignores_future(self):
  d=self.data();c=Config(10,100);n=initial_shards(d,c,474,1)
  d['loads'][474:]=999999
  self.assertEqual(n,initial_shards(d,c,474,1))
 def test_preparation_charged_with_legal_ramp(self):
  d=self.data();c=Config(10,100);SPLITS['tiny']=(6,10)
  d['migrations'][:]=1
  for i in range(7):d['migrations'][:,i,i]=0
  try:
   trace,stats=evaluate_policy(d,c,'tiny',lambda s:0,8)
   self.assertEqual(stats['initial_shards'],8);self.assertEqual(stats['preparation_minutes'],90)
   self.assertAlmostEqual(stats['preparation_cost'],33+6*4/100)
   self.assertAlmostEqual(stats['total_with_preparation'],32+33+6*4/100)
  finally:del SPLITS['tiny']
 def test_preparation_overload_not_hidden(self):
  d=self.data();d['loads'][:6]=100;c=Config(10,100);SPLITS['tiny']=(6,10)
  try:
   _,stats=evaluate_policy(d,c,'tiny',lambda s:0,8)
   self.assertEqual(stats['overloaded_slots'],0);self.assertEqual(stats['preparation_overloaded_slots'],6)
  finally:del SPLITS['tiny']
