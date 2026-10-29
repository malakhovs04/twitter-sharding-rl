import unittest,tempfile,subprocess
from pathlib import Path
import numpy as np
from .prepare import prepare,load
from .environment import Config
from .train import fit

class PipelineTests(unittest.TestCase):
    def test_exact_inventory_and_write_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);raw=b'0,a,2,8,0,set,900\n1,a,2,3,0,get,0\n900,a,2,18,0,set,0\n625000,a,2,3,0,get,0\n'
            result=subprocess.run(['zstd','-q','-c'],input=raw,check=True,stdout=subprocess.PIPE)
            source=p/'source.zst';source.write_bytes(result.stdout);prepare(source,p/'out');d=load(p/'out')
            self.assertEqual(d['inventory'][0,0],0)
            self.assertEqual(d['inventory'][1,0],0) # expiry before boundary's new write
            self.assertEqual(d['inventory'][2,0],20)
            self.assertEqual(d['writes'][0,0],10) # get bytes do not enter dirty log
            self.assertEqual(d['work'][0,0],3+15/4096)
            self.assertEqual(d['hashes'].dtype,np.uint64)
    def test_training_does_not_access_later_windows(self):
        from .test_environment import PhysicsTests
        d=PhysicsTests().data();p={'episodes':1,'development_training_windows':[[0,2],[0,2],[0,2]]}
        hp={'gamma':.9,'rate':'low'};c=Config(10,25,100)
        for kind in ('q','dqn_matched','dqn_continuous'):
            a,_,_=fit(d,c,hp,7,kind,p)
            d['work'][2:]=1e8;d['inventory'][3:]=1e8
            b,_,_=fit(d,c,hp,7,kind,p)
            if kind=='q':
                self.assertEqual(a.keys(),b.keys())
                for k in a:np.testing.assert_array_equal(a[k],b[k])
            else:
                for x,y in zip(a.p,b.p):np.testing.assert_array_equal(x,y)
    def test_unknown_operations_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);raw=b'0,a,2,8,0,append,0\n625000,a,2,3,0,get,0\n'
            result=subprocess.run(['zstd','-q','-c'],input=raw,check=True,stdout=subprocess.PIPE)
            source=p/'source.zst';source.write_bytes(result.stdout)
            with self.assertRaisesRegex(ValueError,'Unsupported operations'):prepare(source,p/'out')
if __name__=='__main__':unittest.main()
