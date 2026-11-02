"""Rerun selected curriculum notebooks from the repository in fresh kernels."""
import argparse,json,os,time
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('notebooks',nargs='+',help='Repository-relative notebook paths')
args=parser.parse_args()
for name in args.notebooks:
    path=(ROOT/name).resolve()
    if not path.is_relative_to(ROOT) or path.suffix!='.ipynb': raise SystemExit('Expected a notebook within this repository')
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    start=time.monotonic()
    try:
        NotebookClient(notebook,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        nbformat.write(notebook,path)
    print(f'PASS {name} ({time.monotonic()-start:.1f} seconds)')
