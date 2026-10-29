import unittest
import numpy as np
from guard_diagnostic import admit
from v2.environment import Env,Config
import v2.test_environment as fixtures
class GuardTests(unittest.TestCase):
    def test_rejects_unsustainable_copy(self):
        d=fixtures.PhysicsTests().data();d['writes'][:4,0]=30;e=Env(d,Config(10,25,100),4,8)
        self.assertEqual(admit(e,1)[:2],(0,True))
    def test_allows_feasible_copy(self):
        d=fixtures.PhysicsTests().data();e=Env(d,Config(10,25,100),4,8)
        self.assertEqual(admit(e,1)[:2],(1,False))
    def test_future_writes_cannot_change_decision(self):
        d=fixtures.PhysicsTests().data();e=Env(d,Config(10,25,100),4,8);a=admit(e,1)
        d['writes'][4:]=1e9;self.assertEqual(a,admit(e,1))
if __name__=='__main__':unittest.main()
