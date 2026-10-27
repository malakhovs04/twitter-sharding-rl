import unittest,tempfile
import numpy as np
from data_pipeline import Inventory,jch,key_hash
from simulator import Env,Config,calibrate
from agents import MLP,double_targets,train_dqn
class Tests(unittest.TestCase):
    def data(self):return dict(loads=np.zeros((672,7,8),dtype=np.int64),migrations=np.zeros((673,7,7),dtype=np.int64),live=np.zeros(673,dtype=np.int64))
    def test_uint64(self):
        for k in (0,1,2**53+7,2**64-1):
            for n in range(2,9):
                a,b=jch(k,n-1),jch(k,n);self.assertTrue(a==b or b==n-1)
        for k in (-1,2**64,1.0):
            with self.assertRaises(ValueError):jch(k,2)
        self.assertEqual(key_hash('abc'),key_hash('abc'))
    def test_inventory_overwrite_expiry_delete(self):
        inv=Inventory();mapping=tuple(jch(key_hash('x'),n) for n in range(2,9))
        inv.write('x',10,10,0,mapping);inv.write('x',20,30,5,mapping)
        inv.expire(10);self.assertEqual(inv.bytes,20)
        expected=(np.array(mapping)[:,None]!=np.array(mapping)[None,:])*20
        np.testing.assert_array_equal(inv.matrix,expected)
        inv.expire(35);self.assertEqual(inv.bytes,0);self.assertFalse(inv.matrix.any())
        inv.write('x',8,0,36,mapping);inv.expire(99999);self.assertEqual(inv.bytes,8)
        inv.remove('x');self.assertEqual(inv.bytes,0)
    def test_accumulated_repeated_move_cost(self):
        d=self.data();d['migrations'][1:,0,1]=100;d['migrations'][1:,1,0]=100
        e=Env(d,Config(10,100),0,3);e.step(0)
        row=e.step(1)[3];self.assertEqual(row['moved_bytes'],100);self.assertEqual(row['cost'],7)
        self.assertEqual(e.step(-1)[3]['moved_bytes'],100)
    def test_causal_observation(self):
        d=self.data();a=Env(d,Config(10,100),0,384).observe().copy()
        d['loads'][:]=999;d['live'][1:]=999;d['migrations'][1:]=999
        np.testing.assert_array_equal(a,Env(d,Config(10,100),0,384).observe())
    def test_train_only_calibration(self):
        d=self.data();a=calibrate(d);d['loads'][384:]=999999;d['live'][384:]=999999
        self.assertEqual(a,calibrate(d))
    def test_terminal_and_double_target(self):
        actual=double_targets(np.array([2.,2.]),np.array([1,0]),np.array([[1,3,2],[1,3,2]]),np.array([[8,4,9],[8,4,9]]),.5)
        np.testing.assert_allclose(actual,[2,4])
    def test_gradient_and_optimizer(self):
        n=MLP(8,4);x=np.array([[.2]*10,[.7]*10]);a=np.array([0,2]);y=np.array([.1,-.2]);before,g=n.gradients(x,a,y)
        for layer,index in [(0,(0,0)),(1,(0,)),(2,(0,0)),(3,(0,)),(4,(0,0)),(5,(0,))]:
            old=n.p[layer][index];eps=1e-6
            n.p[layer][index]=old+eps;p=n.gradients(x,a,y)[0]
            n.p[layer][index]=old-eps;m=n.gradients(x,a,y)[0];n.p[layer][index]=old
            self.assertAlmostEqual(g[layer][index],(p-m)/(2*eps),places=6)
        for _ in range(100):n.update(x,a,y,.001)
        self.assertLess(n.gradients(x,a,y)[0],before)
    def test_save(self):
        n=MLP(8)
        with tempfile.TemporaryDirectory() as t:
            n.save(t+'/model.npz');np.testing.assert_array_equal(n.forward(np.ones(10)),MLP.load(t+'/model.npz').forward(np.ones(10)))
    def test_training_ignores_holdout(self):
        d=self.data();c=Config(10,100);hp=dict(hidden=8,lr=.001,gamma=.95,target_interval=200)
        # Validation may select checkpoints, so use one final checkpoint only.
        a,_=train_dqn(d,c,hp,episodes=1,checkpoint=1)
        d['loads'][480:]=999999;d['migrations'][481:]=999999
        b,_=train_dqn(d,c,hp,episodes=1,checkpoint=1)
        for x,y in zip(a.p,b.p):np.testing.assert_array_equal(x,y)
    def test_env_bounds(self):
        e=Env(self.data(),Config(10,100),0,1)
        self.assertEqual(e.step(-1)[3]['shards'],2)
        with self.assertRaises(ValueError):e.step(0)
if __name__=='__main__':unittest.main()
