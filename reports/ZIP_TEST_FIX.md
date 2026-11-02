# Tests in downloaded ZIP archives

The notebook validation test previously called `git ls-files` to select notebooks. A downloaded ZIP contains no `.git` directory, so this caused `CalledProcessError` with Git exit code 128 even though the notebooks were valid.

The test now validates the daily notebooks directly from their project paths, without calling Git or skipping untracked files. It still checks notebook structure, execution counts and saved error outputs.

Verification on 2 November 2026:

- Reproduced the failure in a clean extraction of the delivered ZIP.
- After the fix, the extracted ZIP passed the full suite: **85 passed**, one upstream deprecation warning.
- The Git checkout passed the three affected contract tests. Its full suite also passed before this isolated test-only change.

Run `python -m pytest -q` from the folder containing `app.py`, using the project environment described in `docs/REPRODUCIBILITY.md`.
