# Loan Approval Prediction

Five classifiers compared under fold-contained preprocessing, nested validation
and SHAP interpretation. Author: Victor Okeke.

[Explore the case study](https://victorokeke.co.uk/projects/loan-approval-prediction/)

## Problem and dataset

Predict the recorded Approved / Rejected decision from 11 application attributes.
The supplied CSV contains 4,269 applications, 13 original columns and no duplicate
records: 2,656 Approved and 1,613 Rejected. Five columns have missing values.
The target describes approval policy; it is not repayment or default risk.

The source report attributes the dataset to Kaggle but does not identify an exact
dataset version or licence. This source release deliberately excludes the CSV.
Place the supplied `loan_approval.csv` in `data/`; its expected SHA-256 is recorded
in `outputs/metrics/dataset.json`. Do not infer collection provenance or a
redistribution licence from the dataset's familiar title.

## Preprocessing and class imbalance

Numeric mean imputation and standardisation; categorical mode imputation and
one-hot encoding. SMOTE runs only inside fitting folds. The ID and target never
enter the predictor matrix. See [methodology](docs/methodology.md) for boundaries
and the categorical-interpolation limitation.

## Models and validation

Logistic Regression, Random Forest, XGBoost, Gradient Boosting and SVC share an
80/20 stratified split. Model selection uses five outer training-only folds with
five inner tuning folds. The selected family is frozen before holdout evaluation.
Default and tuned models are compared on the same outer folds. Metrics include
accuracy, precision, recall, F1, ROC AUC and average precision.

## Results and explainability

Machine-readable results: `outputs/metrics/results.json`; model-selection record:
`selection.json`; SHAP importance and additivity check: `shap.json`.
Figures include unified comparison, CV F1, tuning, confusion matrix, ROC, PR and
SHAP. SHAP explains the model's Approved-class probability, not causal effects.
The model card records the selected family and verified results.

## Reproduce

Use Python 3.11 in a fresh environment:

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
python -m src.train
python -m src.figures
python -m src.explain
python -m src.verify
```

Training performs nested grid searches and can take several minutes. Output paths
are resolved relative to the package. Seed 42 is used throughout. Reload
`models/final_pipeline.joblib` and pass a dataframe with the original predictor
column names; preprocessing runs automatically and SMOTE is skipped at inference.
Only load your own trusted joblib artefacts, using the recorded environment.
`requirements-lock.txt` captures the complete environment used for this run;
install it instead of `requirements.txt` for the closest dependency reproduction.

## Ethical considerations and limitations

Historical approvals may reflect previous policy and proxy bias. Predictive
accuracy and SHAP do not establish fairness. No protected-group assessment,
repayment labels, loss estimates, calibration study or external validation is
available. This research model is inappropriate for automated lending decisions.
Further work: categorical-aware resampling, threshold and calibration analysis,
external/temporal testing and a documented fairness assessment with human oversight.

## Structure

`src/`: training, preprocessing, evaluation, figures, explanation and verification.
`notebooks/`: reproducible analysis entry point. `docs/`: method and model card.
`outputs/`: aggregate evidence. `models/`: locally generated complete pipeline.
Virtual environments, academic administration and redundant saved models are
excluded from the source archive.
