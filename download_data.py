"""Download public source over HTTPS; no full multi-TB dataset is needed."""
from pathlib import Path
import urllib.request,subprocess
from data_pipeline import SOURCE
path=Path('data/cluster7.sort.sample100.zst');path.parent.mkdir(exist_ok=True)
if not path.exists():
    tmp=path.with_suffix('.part')
    urllib.request.urlretrieve(SOURCE,tmp);tmp.rename(path)
subprocess.run(['zstd','-t',str(path)],check=True)
print(path.resolve())
