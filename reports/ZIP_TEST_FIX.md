# Tests in downloaded ZIP archives

The notebook validation test previously called `git ls-files` to select notebooks. A downloaded ZIP contains no `.git` directory, so this caused `CalledProcessError` with Git exit code 128 even though the notebooks were valid.

The test now validates the daily notebooks directly from their project paths, without calling Git or skipping untracked files. It still checks notebook structure, execution counts and saved error outputs.

Verification on 2 November 2026:

- Reproduced the failure in a clean extraction of the delivered ZIP.
- After the fix, the extracted ZIP passed the full suite: **85 passed**, one upstream deprecation warning.
- The Git checkout passed the three affected contract tests. Its full suite also passed before this isolated test-only change.

Run `python -m pytest -q` from the folder containing `app.py`, using the project environment described in `docs/REPRODUCIBILITY.md`.

A subsequent check of the current remote found commit `28a285d` (Organize notebooks and results), which moved evaluation CSVs and early notebooks into `notebooks/`. Before path fixes, that layout reproduced 6 failures and 10 errors. The dashboard, explainability loader and final evaluation now locate results in the organized directory while supporting the original layout. The notebook index now finds all 45 current paths. Three added regression tests cover result discovery and collection completeness.

After these fixes the complete suite passed: **88 passed**, four upstream deprecation warnings, no failures or errors.
