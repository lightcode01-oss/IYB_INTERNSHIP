"""Map preserved original names and new notebooks to curriculum days."""
import re
from urllib.parse import quote
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def notebook_index():
    entries={}
    for day,name in [(1,'YBI_45_Days_Student_Performance_Prediction.ipynb'),(2,'YBI_45_Days_Student_Performance_Prediction(DAY02).ipynb')]:
        candidates=[ROOT/'notebooks'/name,ROOT/name]
        path=next((p for p in candidates if p.is_file()),None)
        if path is None: raise FileNotFoundError(name)
        entries[day]=path.relative_to(ROOT).as_posix()
    for p in [*ROOT.glob('Day_*.ipynb'),*(ROOT/'notebooks').glob('Day_*.ipynb')]:
        match=re.match(r'Day_(\d+)_',p.name)
        if match:
            day=int(match.group(1))
            if day in entries: raise ValueError(f'Duplicate daily notebook: {day}')
            entries[day]=p.relative_to(ROOT).as_posix()
    return dict(sorted(entries.items()))

def write_index():
    entries=notebook_index()
    lines=['# Daily notebook index','','Days are curriculum milestones. Paths below follow the current repository layout; the original notebook filenames are preserved.','','| Day | Notebook |','|---|---|']
    for day,path in entries.items(): lines.append(f'| {day:02d} | [{Path(path).stem}](../{quote(path,safe="/")}) |')
    (ROOT/'docs/NOTEBOOK_INDEX.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return entries

if __name__=='__main__': print('Indexed',len(write_index()),'daily notebooks')
