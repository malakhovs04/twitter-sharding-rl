import unittest
from dataclasses import replace
import numpy as np
from .environment import Env,Config,calibrate,encode,state_key

class PhysicsTests(unittest.TestCase):
    def data(self):
        d=dict(work=np.zeros((673,2)),writes=np.zeros((673,2)),inventory=np.zeros((674,2)),mapping=np.zeros((7,2),dtype=int))
        d['mapping'][:,1]=1;d['mapping'][3:,0]=2;d['inventory'][:,0]=100
        return d
    def config(self):return Config(10,25,100,migration_cpu_fraction=.2)
    def test_finite_migration_and_billing(self):
        e=Env(self.data(),self.config(),0,8,initial=4)
        rows=[e.step(1)[3] for _ in range(4)]
        self.assertEqual([r['shards'] for r in rows],[4,4,4,5])
        self.assertEqual(sum(r['transferred_bytes'] for r in rows),100)
        self.assertEqual([r['billed_shards'] for r in rows],[5]*4)
        self.assertEqual(sum(r['migration_started'] for r in rows),1)
    def test_dirty_log_conservation(self):
        d=self.data();d['writes'][:3,0]=10;e=Env(d,self.config(),0,8)
        for _ in range(3):
            row=e.step(1)[3]
            self.assertAlmostEqual(row['transfer_before']+row['dirty_bytes'],row['transferred_bytes']+row['transfer_remaining'])
        self.assertEqual(e.remaining,55)
    def test_queue_conservation_at_cutover(self):
        d=self.data();d['work'][:8,0]=100;e=Env(d,self.config(),0,8)
        for _ in range(8):
            r=e.step(1)[3]
            self.assertAlmostEqual(r['queue_before']+r['incoming_work'],r['served_work']+r['queue_after'])
        self.assertGreater(e.queue.sum(),0)
    def test_migration_reduces_service(self):
        d=self.data();d['work'][0,0]=10;a=Env(d,self.config(),0,2);b=Env(d,self.config(),0,2)
        self.assertGreater(a.step(1)[3]['queue_after'],b.step(0)[3]['queue_after'])
    def test_holdout_future_not_observed(self):
        d=self.data();a=Env(d,self.config(),96,192).observe()
        d['work'][96:]=1e9;d['writes'][96:]=1e9;d['inventory'][97:]=1e9
        np.testing.assert_array_equal(a,Env(d,self.config(),96,192).observe())
    def test_calibration_train_only(self):
        d=self.data();a=calibrate(d);d['work'][384:]=999999;d['inventory'][384:]=999999
        self.assertEqual(a,calibrate(d))
    def test_terminal_liabilities_not_deleted(self):
        d=self.data();d['work'][0,0]=100;e=Env(d,self.config(),0,1)
        r=e.step(1)[3];self.assertGreater(r['terminal_queue_cost'],0);self.assertGreater(r['terminal_transfer_cost'],0)
    def test_matched_encoding_contains_exact_same_bins(self):
        for s in [np.zeros(10),np.arange(10)/4,np.ones(10)*100]:
            from .environment import EDGES
            decoded=tuple(int(round(x*(len(e)+1))) for x,e in zip(encode(s),EDGES))
            self.assertEqual(decoded,state_key(s))
    def test_hot_key_not_fixed_by_more_shards(self):
        d=self.data();d['work'][0,0]=100
        for n in range(2,9):self.assertEqual(Env(d,self.config(),0,1,n).step(0)[3]['hot_key_over_capacity'],1)
    def test_bandwidth_changes_duration(self):
        for bandwidth,expected in [(25,4),(50,2),(100,1)]:
            e=Env(self.data(),replace(self.config(),bandwidth=bandwidth),0,8)
            for step in range(expected):r=e.step(1)[3]
            self.assertEqual(r['shards'],5)
    def test_zero_copy_still_switches_only_after_service(self):
        d=self.data();d['inventory'][:]=0;d['work'][0,0]=100;e=Env(d,self.config(),0,2)
        r=e.step(1)[3];self.assertEqual(r['old_shards'],4);self.assertEqual(r['shards'],5);self.assertEqual(r['served_work'],10)

if __name__=='__main__':unittest.main()
