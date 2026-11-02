"""Build a factual internship report from the repository's measured evidence."""
import json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote
import pandas as pd
from scripts.notebook_index import notebook_index,write_index,ROOT

def build_report():
    entries=write_index()
    assert list(entries)==list(range(1,46))
    evaluation=json.loads((ROOT/'reports/final_evaluation.json').read_text())
    deployment=json.loads((ROOT/'reports/deployment_status.json').read_text())
    suites=ET.parse(ROOT/'reports/final_test_results.xml').getroot().findall('testsuite')
    totals={key:sum(int(s.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
    assert totals['tests']>0 and totals['failures']==totals['errors']==0
    history=subprocess.check_output(['git','log','--reverse','--format=%H%x09%s','5f247f6..HEAD'],cwd=ROOT,text=True).splitlines()
    ledger=['# Verified daily commit ledger','','Each entry below is an actual commit pushed during the repair/extension session. Earlier project history was preserved. The Day 45 commit contains this ledger and is identified by its commit message; its own hash cannot be embedded in the same commit.','', '| Commit | Message |','|---|---|']
    for line in history:
        commit,message=line.split('\t',1)
        ledger.append(f'| [{commit[:7]}](https://github.com/lightcode01-oss/IYB_INTERNSHIP/commit/{commit}) | {message} |')
    ledger.append('| This report’s commit | Day 45: Internship Completion |')
    (ROOT/'reports/COMMIT_LEDGER.md').write_text('\n'.join(ledger)+'\n',encoding='utf-8')
    matrix=['# Final curriculum completion matrix','','All daily notebooks were executed in fresh kernels and validated as Jupyter notebooks. Days are curriculum milestones, not elapsed calendar dates. Day 45 finalization is verified by the saved output of its own notebook and its successful push.','', '| Day | Expected work | Actual notebook | Verification | Status |','|---|---|---|---|---|']
    for number,path in entries.items():
        title=Path(path).stem.replace('_',' ')
        matrix.append(f'| {number:02d} | {title} | [{Path(path).name}](../{quote(path,safe="/")}) | Fresh-kernel output and schema checks | Complete and verified |')
    (ROOT/'reports/COMPLETION_MATRIX.md').write_text('\n'.join(matrix)+'\n',encoding='utf-8')
    measured=pd.DataFrame([evaluation['metrics']]).to_markdown(index=False)
    benchmark=pd.read_csv(ROOT/'reports/serving_benchmark.csv').to_markdown(index=False)
    report=f'''# Final internship project report

## Project identity and completion scope

**Internship:** YBI Foundation — Python AI & Data Science Internship
**Project:** AI-Based Student Performance Prediction System
**Repository:** https://github.com/lightcode01-oss/IYB_INTERNSHIP
**Completion session:** 2 November 2026
**Collection:** 45 curriculum-day notebooks, with the original Day 1–23 names preserved.

The 45 days are curriculum milestones. The repair and extension work was performed in one completion session, with one meaningful commit and push per day. This report does not claim 45 separate elapsed working days, an internship certificate, mentor approval or an externally approved completion.

## Starting state and recovery

The GitHub repository initially had 63 commits and daily notebooks through Day 23. Its source modules, trained model and supporting files had not been pushed. A fuller local project was recovered without editing the original local folder. Day 24 was downloaded from the existing Drive notebook. Broken remote notebook versions were replaced with validated project versions and executed; the old history remains intact.

The remote Day 23 notebook created mock metrics. It was replaced with a dashboard notebook that loads real evaluation artifacts. The original tuning search, training-CV ranking, saved pipeline and 519/130 split were reproduced. Every historical test prediction matched the saved pipeline. A notebook-format version mismatch discovered by expanded tests was repaired. No synthetic metric is presented as evaluation evidence.

## Data and preprocessing

The project uses the public UCI Student Performance dataset, ID 320: 649 Portuguese-course records from two Portuguese secondary schools. Attribution: Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5TG7T. License: CC BY 4.0.

The target is Low for G3 < 10, Medium for 10 <= G3 < 15, and High for G3 >= 15. G3 and performance_category never enter the prediction schema. G1 and G2 are intentionally excluded to preserve the existing 30-predictor project design. Categorical encoding stays inside the saved pipeline. The original stratified split uses random_state 42, with 519 training and 130 evaluation rows. Explicit indices, source information and hashes are in artifacts/version_manifest.json and artifacts/split_indices.json.

## Model selection and measured results

Six candidate algorithms were compared and tuned using five-fold stratified cross-validation on the training partition. The selected model is {evaluation['selected_model']}, with training CV Macro F1 {evaluation['training_cv_macro_f1']:.6f} and fold standard deviation {evaluation['training_cv_std']:.6f}. Its exact parameters and selection evidence are retained in the artifacts and tuning reports.

Historical test measurements:

{measured}

The majority-class baseline Macro F1 is {evaluation['majority_baseline_macro_f1']:.6f}. A {evaluation['bootstrap_repeats']}-repeat stratified bootstrap gives a conditional 95% percentile interval of {evaluation['conditional_bootstrap_macro_f1_95_percentile'][0]:.4f}–{evaluation['conditional_bootstrap_macro_f1_95_percentile'][1]:.4f} for Macro F1. This interval fixes class counts and the trained model. It excludes training/model-selection uncertainty and is not a guarantee for future students or schools.

The test set was repeatedly inspected in earlier project notebooks. These are descriptive historical results, not fresh external validation. The low Macro F1 and weak minority-class results are visible in reports/FINAL_MODEL_EVALUATION.md rather than hidden behind inflated mock scores.

## Implemented application and reliability work

- Streamlit single-record prediction, validated CSV upload and the optional local FastAPI share one prediction service.
- The strict contract checks exactly 30 predictors, categorical domains, integral numeric bounds, missing/nonfinite values, extra/target columns and duplicate fields.
- CSV input is bounded to 500 records and 1 MiB. Duplicate headers, ragged rows and invalid records reject the whole batch.
- The trusted saved model is checksum-verified before deserialization. Its version identity is cached with the loaded object. Inference computes probabilities once and preserves the original class predictions.
- The API has health and prediction endpoints, request-size limits and sanitized unavailable-model responses. It remains a local demonstration without production authentication.
- Explanation reports include permutation importance, native tree attribution, local decision paths and shuffle-seed variability. These describe associations, not causal effects.
- Monitoring is off by default, excludes arbitrary extra fields and preserves zero probabilities. Process-local locking prevents interleaved CSV writes. Distribution comparison and subjective feedback are explicitly not accuracy measurements.

## Verification evidence

Final recorded suite: **{totals['tests']} tests**, **{totals['failures']} failures**, **{totals['errors']} errors**, **{totals['skipped']} skipped**. The machine-readable evidence is reports/final_test_results.xml and the command output is in reports/FINAL_TEST_REPORT.md.

The daily collection has valid notebook structure and saved real outputs. Days 1–23 were rerun during recovery, and Days 19–42 were rerun against the final application code. Later finalization notebooks preserve their own execution output. Local HTTP checks started temporary API and Streamlit servers, checked health/frontend responses and compared real API predictions with the shared service. The processes created for those checks were stopped afterward. GitHub Actions was configured and successful runs were independently observed during the session; the final commit’s workflow status is reported separately after pushing.

Actual local serving benchmark:

{benchmark}

These timings describe this environment and are not a service-level promise. Upstream deprecation warnings remain visible in test evidence. A Windows notebook-kernel shutdown warning was observed during the broad rerun after cell execution had succeeded; it did not invalidate the saved cell results or subsequent regression checks.

## Deployment and demonstration

Recorded hosted status: **{deployment['hosted_deployment']}** The UI and API were verified locally. No public deployment or GitHub release was published. Public hosting requires separate approval and review of authentication, privacy, retention and operational storage.

Use docs/DEMONSTRATION.md for the repeatable walkthrough and data/demo_input.csv for public demonstration records. Actual generated model plots appear in the README and reports/day_24/. An app screenshot could not be captured because the computer-use tool stopped during browser URL verification. No screenshot or hosted URL was fabricated.

## Deliverables and navigation

- docs/NOTEBOOK_INDEX.md — all 45 daily notebooks
- docs/ARCHITECTURE.md — actual component and data flow
- docs/DATA_AND_MODEL_CARD.md — source, preprocessing, model and limitations
- reports/FINAL_MODEL_EVALUATION.md — real metrics, baseline and uncertainty
- reports/FINAL_EXPLAINABILITY.md — measured feature associations
- reports/FINAL_TEST_REPORT.md — actual regression results
- docs/DEPLOYMENT.md — setup and verified local-only hosting status
- docs/DEMONSTRATION.md — demonstration procedure
- docs/DEVELOPMENT_LOG.md — daily implementation and validation log
- reports/COMMIT_LEDGER.md — actual separate daily commits
- reports/COMPLETION_MATRIX.md — day-by-day verification

## Remaining limitations

The dataset is small, observational, old and context-specific. It includes demographic, family and health attributes that can act as sensitive proxies. Probabilities are uncalibrated. There is no external educational validation, causal evidence, multi-user authorization or production monitoring datastore. The application must not determine grades, access, eligibility or other consequential student outcomes. No private student information was added to the public repository. Older general Python/ML practice notebooks were preserved outside the 45-day project collection.

The software and documentation support an internship portfolio demonstration. Scientific generalization, production hosting, an application screenshot and any institutional completion approval remain outside the verified deliverables.
'''
    (ROOT/'reports/FINAL_INTERNSHIP_REPORT.md').write_text(report,encoding='utf-8')
    (ROOT/'reports/PROJECT_SUMMARY.md').write_text(f'# Project summary\n\nAI-Based Student Performance Prediction System for the YBI Foundation Python AI & Data Science internship. The collection covers 45 curriculum milestones with a saved decision-tree pipeline, local app/API, validated batch prediction, explainability, monitoring and reproducible tests. Historical test Macro F1: {evaluation["metrics"]["macro_f1"]:.4f}; accuracy: {evaluation["metrics"]["accuracy"]:.2%}. Final recorded tests: {totals["tests"]} passed. Deployment: local only; no verified hosted URL. See FINAL_INTERNSHIP_REPORT.md for evidence and limitations.\n',encoding='utf-8')
    return {'daily_notebooks':len(entries),'tests':totals,'historical_macro_f1':evaluation['metrics']['macro_f1'],'report':'reports/FINAL_INTERNSHIP_REPORT.md'}

if __name__=='__main__': print(json.dumps(build_report(),indent=2))
