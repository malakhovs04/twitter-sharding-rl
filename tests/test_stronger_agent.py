import unittest,itertools,tempfile
import numpy as np
from pathlib import Path
from stronger_agent import Network,RichEnv,teacher_actions,build_teacher_dataset
from simulator import Env,Config
from agents import MLP
class StrongerTests(unittest.TestCase):
 def data(self):
  d=dict(loads=np.zeros((672,7,8),dtype=np.int64),migrations=np.zeros((673,7,7),dtype=np.int64),live=np.zeros(673,dtype=np.int64))
  for i,n in enumerate(range(2,9)):d['loads'][:,i,0]=17
  return d
 def test_expansion_preserves_warm_start(self):
  old=MLP(64,7);net=Network.expand(old,96);rng=np.random.default_rng(2);x=rng.normal(size=(10,52))
  np.testing.assert_allclose(net.forward(x),old.forward(x[:,:10]),atol=1e-10)
 def test_rich_observation_no_current_or_future_demand(self):
  d=self.data();c=Config(10,100);a=RichEnv(d,c,100,101).observe()
  d['loads'][100:]=999999;d['migrations'][101:]=999999;d['live'][101:]=999999
  np.testing.assert_array_equal(a,RichEnv(d,c,100,101).observe());self.assertEqual(len(a),52)
 def test_teacher_ignores_validation_and_test(self):
  d=self.data();c=Config(10,100);x,y=build_teacher_dataset(d,c,factors=(1.,))
  d['loads'][384:]=999999;d['migrations'][385:]=999;d['live'][385:]=999
  xx,yy=build_teacher_dataset(d,c,factors=(1.,));np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(y,yy)
 def test_teacher_matches_exhaustive_search(self):
  d=self.data();c=Config(10,100);labels=teacher_actions(d,c,end=3)
  best=[]
  for actions in itertools.product((-1,0,1),repeat=3):
   env=Env(d,c,0,3);rows=[env.step(a)[3] for a in actions]
   best.append(((sum(r['excess']>0 for r in rows),sum(r['excess'] for r in rows),sum(r['cost'] for r in rows)),actions))
  expected=min(best,key=lambda r:r[0])[0];env=Env(d,c,0,3);rows=[]
  while env.t<3:rows.append(env.step(int(labels[env.t,env.n-2])-1)[3])
  self.assertEqual(expected,(sum(r['excess']>0 for r in rows),sum(r['excess'] for r in rows),sum(r['cost'] for r in rows)))
 def test_cross_entropy_and_q_gradients(self):
  net=Network(8,4);x=np.random.default_rng(1).normal(size=(4,52));y=np.array([0,1,2,0]);w=np.ones(3)
  for mode in ('classification','q'):
   f=(lambda:net.classification_gradient(x,y,w)) if mode=='classification' else (lambda:net.q_gradient(x,y,np.ones(4)))
   loss,g=f()
   for layer,index in [(0,(0,0)),(1,(0,)),(2,(0,0)),(3,(0,)),(4,(0,0)),(5,(0,))]:
    old=net.p[layer][index];epsilon=1e-6;net.p[layer][index]=old+epsilon;p=f()[0];net.p[layer][index]=old-epsilon;m=f()[0];net.p[layer][index]=old
    self.assertAlmostEqual(g[layer][index],(p-m)/(2*epsilon),places=5)
 def test_save_restore(self):
  net=Network(8)
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'model.npz';net.save(p);np.testing.assert_array_equal(net.forward(np.ones(52)),Network.load(p).forward(np.ones(52)))
