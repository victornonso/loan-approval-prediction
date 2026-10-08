from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

NUMERIC = ['income_annum','loan_amount','loan_term','cibil_score','no_of_dependents','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']
CATEGORICAL = ['education','self_employed']
SEED = 42

def preprocessing():
    return ColumnTransformer([
        ('numeric',SkPipeline([('impute',SimpleImputer(strategy='mean')),('scale',StandardScaler())]),NUMERIC),
        ('categorical',SkPipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CATEGORICAL)
    ],verbose_feature_names_out=False)

def pipeline(model):
    # GridSearchCV clones and fits this entire pipeline inside each training fold.
    return Pipeline([('preprocess',preprocessing()),('sampling',SMOTE(random_state=SEED,k_neighbors=5)),('classifier',model)])
