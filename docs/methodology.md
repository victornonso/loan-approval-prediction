# Validation methodology

The source notebook supplied five classifiers, a stratified holdout, grid search,
SMOTE and SHAP. This reproducible edition retains the model families and feature
definitions while correcting the validation boundaries.

1. Strip whitespace in headers and categorical values. Exclude `loan_id` and
   `loan_status` from predictors. Approved is the positive class.
2. Reserve a stratified 20% holdout with seed 42: 854 original applications.
3. Use five stratified outer folds on the remaining 3,415 applications only.
4. In each outer fold, tune through five inner folds. Every candidate contains
   mean numeric imputation, mode categorical imputation, numeric standardisation,
   one-hot encoding and SMOTE in one imbalanced-learn pipeline. Each inner fit
   learns these steps on its own training fold. Validation observations stay real.
5. Select the family with the highest mean outer-fold Approved-class F1. Report
   population standard deviation; lower SD breaks exact ties. Persist this
   selection before inspecting holdout outcomes.
6. Retune each family on all 3,415 training rows. Evaluate all five on the unchanged
   holdout, without changing the selected family. Save the selected full pipeline.
7. Explain the selected model after selection using tree-path-dependent Tree SHAP for Random Forest,
   fitted tree leaf frequencies and all 854 holdout rows. Leaf frequencies include
   SMOTE-generated training observations. Explain
   Approved-class probability; verify additive reconstruction numerically.

The original notebook applied preprocessing and SMOTE before inner grid-search
splits and included the eventual holdout in outer model-selection folds. Its
evaluation figures are retained in the private audit, not reused as corrected
performance evidence. Dataset-only EDA figures remain applicable.

The XGBoost search uses the original notebook's smaller nested-validation grid
consistently for outer and final tuning. No additional models were introduced.
Candidate counts and exact selected parameters are exported to JSON.

Average precision uses `average_precision_score`, rather than trapezoidal PR AUC.
Confusion matrices use rows actual and columns predicted, ordered Rejected,
Approved. Threshold-based metrics use each classifier's default prediction rule.
F1 balances precision and recall for Approved; it does not assess every error cost.

Ordinary SMOTE after one-hot encoding is retained for comparison with the source
study. It can interpolate categorical indicators. Compare SMOTENC, class weighting
and no resampling before any application beyond this study.

Full-dataset EDA is descriptive. It is not an independent validation result.
The original notebook already explored all rows; the revised held-out evaluation
is separated from fitting and family selection, but is not a fresh prospective test.
Model probabilities have not been independently calibrated; no temporal,
external-population or fairness validation has been performed.

References:
- https://imbalanced-learn.org/stable/common_pitfalls.html
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
