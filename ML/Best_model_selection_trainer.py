import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import all_estimators
import joblib

from lazypredict.Supervised import LazyClassifier


def train_and_save_model(df: pd.DataFrame, model_path: str) -> None:
    """
    Train multiple classification models (via LazyPredict), select the best one,
    rebuild a pipeline around that best model, and save it.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and a target column
                           named 'Social_support' (0 or 1). Expected columns (at least):
                - 'declared_income'
                - 'No_of_companies_worked'
                - 'current_employer_tenure_months'
                - 'earliest_start_year'
                - 'total_experience'
                - 'average_employement_tenure_months'
                - 'total_assets'
                - 'total_liabilities'
                - 'net_worth'
                - 'num_assets'
                - 'num_liabilities'
                - 'asset_to_liability_ratio'
                - 'property_asset_value'
                - 'investment_asset_value'
                - 'family_size'
                - 'dependents'
                - 'Age'
                - 'Gender'
                - 'marital_status'
                - 'Disability'
                - 'Housing_type'
                - 'Social_support' (target: 0 or 1)
        model_path (str): File path where the trained model pipeline will be saved
                          (e.g. 'model.pkl').

    Returns:
        None. The trained pipeline is saved to model_path.
    """
    warnings.filterwarnings("ignore")  # suppress any convergence/data warnings

    # 1) Copy & coerce numeric columns
    data = df.copy()
    numeric_cols = [
        'declared_income',
        'No_of_companies_worked',
        'current_employer_tenure_months',
        'earliest_start_year',
        'total_experience',
        'average_employement_tenure_months',
        'total_assets',
        'total_liabilities',
        'net_worth',
        'num_assets',
        'num_liabilities',
        'asset_to_liability_ratio',
        'property_asset_value',
        'investment_asset_value',
        'family_size',
        'dependents',
        'Age'
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # 2) Drop any rows with missing values in either numeric or categorical or target
    data = data.dropna(
        subset=numeric_cols + ['Gender', 'marital_status', 'Disability', 'Housing_type', 'Social_support']
    )

    # 3) Split features/target
    X = data.drop(columns=['Social_support'])
    y = data['Social_support'].astype(int)

    # 4) Identify which columns are numeric vs categorical
    numeric_features = numeric_cols
    categorical_features = ['Gender', 'marital_status', 'Disability', 'Housing_type']

    # 5) Build preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # 6) Do a train/test split (we’ll use the test‐set performance from LazyPredict)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 7) Run LazyPredict to benchmark all classifiers
    print("\n>>> Running LazyClassifier over all scikit‐learn classifiers...\n")
    lc = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        custom_metric=None
    )
    models, predictions = lc.fit(X_train, X_test, y_train, y_test)

    # `models` is a pd.DataFrame indexed by model‐name, sorted by Test Accuracy by default
    print(models)  # show top 10 for reference

    # 8) Identify the single best model by test‐set accuracy (the first index)
    best_model_name = models.index[0]
    print(f"\n>>> Best‐performing classifier according to LazyPredict: {best_model_name}\n")

    # 9) Map that model‐name to its actual class via sklearn.utils.all_estimators
    clf_list = {name: cls for name, cls in all_estimators(type_filter='classifier')}
    if best_model_name not in clf_list:
        raise ValueError(f"Could not find {best_model_name} in all_estimators().")

    BestClassifierClass = clf_list[best_model_name]
    best_clf = BestClassifierClass()

    # 10) Build a new pipeline using the same preprocessor + the best classifier
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', best_clf)
    ])

    # 11) Fit that pipeline on the training data
    final_pipeline.fit(X_train, y_train)

    # 12) Evaluate once more on the hold‐out test set
    y_pred = final_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Re‐trained '{best_model_name}' Pipeline Test Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))

    # 13) Save the pipeline containing the best model
    joblib.dump(final_pipeline, "Models/"+model_path+best_model_name+".pkl")
    print(f"\n>>> Final pipeline (with {best_model_name}) saved to: {model_path}\n")
