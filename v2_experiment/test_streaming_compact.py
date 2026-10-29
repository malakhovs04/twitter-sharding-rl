import unittest,tempfile,subprocess
from pathlib import Path
import numpy as np
from prepare_compact import prepare as streaming
from prepare_offset import prepare as dense
from v2.prepare import load
from compact_data import compact,load_compact
class StreamingTests(unittest.TestCase):
    def test_identical_aggregates_with_ttl_and_add(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);raw=b'1,x,2,8,0,add,900\n2,y,3,9,0,set,0\n3,x,2,98,0,add,0\n899,y,3,7,0,gets,0\n900,x,2,8,0,get,0\n901,x,2,18,0,add,0\n2700,z,4,2,0,get,0\n625000,x,2,8,0,get,0\n'
            result=subprocess.run(['zstd','-q','-c'],input=raw,check=True,stdout=subprocess.PIPE);src=p/'s.zst';src.write_bytes(result.stdout)
            dense(src,p/'dense');streaming(src,p/'small');a=compact(load(p/'dense'));b=load_compact(p/'small')
            indices=[next(j for j,x in enumerate(b['mapping'].T) if np.array_equal(sig,x)) for sig in a['mapping'].T]
            for name in ('work','writes','inventory','counts'):np.testing.assert_array_equal(a[name],b[name][:,indices])
            np.testing.assert_array_equal(a['hot_work'],b['hot_work'])
if __name__=='__main__':unittest.main()
