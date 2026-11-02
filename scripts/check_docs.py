"""Check relative Markdown links in the maintained documentation index."""
import re
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[1]
DOCUMENTS=['docs/INDEX.md','docs/NOTEBOOK_INDEX.md','docs/ARCHITECTURE.md']

def check_docs():
    checked=0
    for relative in DOCUMENTS:
        path=ROOT/relative
        text=path.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('http:','https:','#')): continue
            resolved=path.parent/unquote(target.split('#')[0])
            if not resolved.exists(): raise ValueError(f'Broken link in {relative}: {target}')
            checked+=1
    return checked

if __name__=='__main__': print('Relative documentation links checked:',check_docs())
