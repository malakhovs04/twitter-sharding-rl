import unittest,tempfile,subprocess
from pathlib import Path
from prepare_offset import prepare
from v2.prepare import load
class OffsetTest(unittest.TestCase):
    def test_one_second_offset_preserves_timestamp(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);raw=b'1,x,2,8,0,set,0\n625000,x,2,8,0,get,0\n'
            result=subprocess.run(['zstd','-q','-c'],input=raw,check=True,stdout=subprocess.PIPE)
            src=p/'s.zst';src.write_bytes(result.stdout);prepare(src,p/'d');d=load(p/'d')
            self.assertEqual(d['counts'][0,0],1);self.assertEqual(d['inventory'][0,0],0);self.assertEqual(d['inventory'][1,0],10)
if __name__=='__main__':unittest.main()

class AddSemanticsTest(unittest.TestCase):
    def test_add_existing_does_not_overwrite_or_add_dirty_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);raw=b'1,x,2,8,0,add,900\n2,x,2,98,0,add,0\n3,x,2,8,0,gets,0\n901,x,2,18,0,add,0\n625000,x,2,8,0,get,0\n'
            result=subprocess.run(['zstd','-q','-c'],input=raw,check=True,stdout=subprocess.PIPE)
            src=p/'s.zst';src.write_bytes(result.stdout);prepare(src,p/'d');d=load(p/'d')
            self.assertEqual(d['inventory'][1,0],10)
            self.assertEqual(d['inventory'][2,0],20)
            self.assertEqual(d['writes'][0,0],10)
            self.assertEqual(d['writes'][1,0],20)
            self.assertEqual(d['work'][0,0],5+120/4096)
