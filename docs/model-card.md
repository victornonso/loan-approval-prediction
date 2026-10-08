# Loan Approval Prediction — model card
Author: Victor Okeke

## Purpose
Research classification of historical Approved / Rejected application labels.
Not a default-risk, repayment or automated lending model.

## Dataset and features
4,269 supplied applications; 2,656 Approved and 1,613 Rejected. 11 predictors
(nine numeric, two categorical); ID and target excluded. Five columns have missing
values. No duplicate rows and unique application IDs. Source report cites Kaggle;
exact source version, licence, collection provenance and decision policy are unknown.
CSV SHA-256: `0c0f1a6ec99db09b548aba665ac3e59e92e1e4afefbb4b3f16897b4778fdef09`.

## Preprocessing and validation
Mean numeric / mode categorical imputation, numeric standardisation, one-hot
encoding and SMOTE are fitted inside each inner training fold. 80/20 stratified
split, seed 42. Five outer training-only folds and five inner tuning folds.
Selected family: **Random Forest** by highest mean outer-fold F1.
CV F1: 0.9540; population SD 0.0038.
Holdout: 854 original rows, unused for family selection.

## Held-out metrics
Approved is positive. Accuracy 0.9567; precision
0.9540; recall 0.9774; F1 0.9656;
ROC AUC 0.9911; average precision 0.9940.
Confusion matrix (actual rows / predicted columns, Rejected then Approved):
`[[298, 25], [12, 519]]`. Metrics use the model's default prediction rule.

## Explanation
Tree-path-dependent Tree SHAP explains Approved-class probability for 854 holdout rows.
Background: Fitted tree leaf frequencies, including SMOTE training observations. Top transformed features:
cibil_score, loan_term, loan_amount.
Maximum additive reconstruction error: 4.55e-15.
Contributions describe the model, not causal effects.

## Limitations and inappropriate use
The source notebook previously explored all rows; this is not a new prospective
test. No repayment outcomes, external/temporal testing, calibration assessment or
protected-group fairness assessment. Proxy bias remains possible. Historical
policy, limited predictors and dataset shift constrain generalisation.
SMOTE interpolates one-hot indicators; compare categorical-aware sampling and
class weights. Accuracy and explainability do not establish fairness.
Do not use this model to approve/reject borrowers, infer creditworthiness or give
lending advice. Any real application needs appropriate governance, fairness and
policy assessment, monitoring and human oversight.

## Reproduction
See README, methodology, environment.json, selection.json and source modules.
The saved local artefact contains all preprocessing and the classifier together.
