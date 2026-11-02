"""Find preserved evaluation results in the organized or legacy layout."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def result_path(filename, root=None):
    base = Path(root) if root is not None else ROOT
    for candidate in (base / 'notebooks' / filename, base / filename):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Evaluation result not found: {filename} (checked notebooks/ and the project root)')
