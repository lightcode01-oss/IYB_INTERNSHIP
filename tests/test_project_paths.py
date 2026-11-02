import pytest
from src.project_paths import result_path
from scripts.notebook_index import notebook_index,ROOT

def test_result_path_organized_and_legacy(tmp_path):
    name='results.csv'
    legacy=tmp_path/name;legacy.write_text('legacy')
    assert result_path(name,tmp_path)==legacy
    organized=tmp_path/'notebooks'/name
    organized.parent.mkdir();organized.write_text('organized')
    assert result_path(name,tmp_path)==organized

def test_result_path_missing(tmp_path):
    with pytest.raises(FileNotFoundError,match='missing.csv'):
        result_path('missing.csv',tmp_path)

def test_all_curriculum_notebook_paths_exist():
    entries=notebook_index()
    assert list(entries)==list(range(1,46))
    assert all((ROOT/path).is_file() for path in entries.values())
