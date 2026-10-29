import unittest
import numpy as np
import pandas as pd
from simulator import Config,Env
from methodology_audit import target_policy,reprice,decision_trace

class AuditTests(unittest.TestCase):
    def test_common_start_and_one_step(self):
        data={'loads':np.zeros((673,7,8),dtype=int),'live':np.zeros(673),'migrations':np.ones((673,7,7),dtype=int)}
        tr=decision_trace(data,Config(10,1),'test',target_policy(8))
        self.assertEqual(tr.old_shards.iloc[0],2)
        self.assertEqual(list(tr.shards[:6]),[3,4,5,6,7,8])
        self.assertEqual(tr.moved_bytes.iloc[0],1)
    def test_reprice_uses_components(self):
        tr=pd.DataFrame(dict(server_cost=[2,3],excess=[10,20],moved_bytes=[100,200]))
        self.assertEqual(reprice(tr,Config(10,100),80,4),257)
    def test_observation_no_current_future_load(self):
        d={'loads':np.ones((673,7,8),dtype=int),'live':np.ones(673),'migrations':np.zeros((673,7,7),dtype=int)}
        before=Env(d,Config(10,100),480,672).observe()
        d['loads'][480:]*=10000
        np.testing.assert_array_equal(before,Env(d,Config(10,100),480,672).observe())
if __name__=='__main__':unittest.main()
