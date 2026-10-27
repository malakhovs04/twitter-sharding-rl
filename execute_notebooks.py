"""Execute all notebooks with the Python environment invoking this script."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import sys
for path in sorted(Path('.').glob('0*.ipynb')):
    nb=nbformat.read(path,as_version=4)
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    try:
        NotebookClient(nb,timeout=1800,km=km).execute()
    finally:
        if km.has_kernel: km.shutdown_kernel(now=True)
    nbformat.write(nb,path)
    print('Executed',path,flush=True)
