from pathlib import Path
import sys
import time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
path = root / 'experiments/China_A_Share_Final_Factor_Timing.ipynb'
notebook = nbformat.read(path, 4)
started = time.monotonic()

def progress(cell, cell_index, **kwargs):
    nbformat.write(notebook, path)
    if cell.cell_type == 'code':
        print(f'Completed cell {cell_index}: {cell.source.splitlines()[0]} ({time.monotonic()-started:.0f}s)', flush=True)

manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(notebook, timeout=1800, km=manager, resources={'metadata': {'path': str(root)}}, on_cell_executed=progress)
try:
    client.execute()
finally:
    nbformat.write(notebook,path)
print('Finished in',round(time.monotonic()-started),'seconds',flush=True)
