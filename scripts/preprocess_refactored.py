# scripts/preprocess_refactored.py
import os, yaml, joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# load config
cfg = yaml.safe_load(open("config.yaml")) if os.path.exists("config.yaml") else {}
local_path = cfg.get("data", {}).get("local_path", "data/raw/raw_data.csv")
preprocessed_path = cfg.get("data", {}).get("preprocessed_path", "data/preprocessed/preprocessed_data.csv")
target_col = cfg.get("data", {}).get("target", "Total.Cup.Points")
test_size = cfg.get("train", {}).get("test_size", 0.2)
random_state = cfg.get("train", {}).get("random_state", 42)

# ensure dirs
os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
os.makedirs("data/cleaned", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# load preprocessed data (created by ingest.py)
df = pd.read_csv(preprocessed_path)
if target_col not in df.columns:
    raise KeyError(f"Target '{target_col}' not found. Columns: {df.columns.tolist()[:20]}")

# lists of columns — ideally put in config, but you can define here
numeric_cols = [
    "Number.of.Bags", "Category.One.Defects", "Category.Two.Defects", "Aroma", "Flavor",
    "Aftertaste", "Acidity", "Body", "Balance", "Uniformity", "Clean.Cup", "Sweetness",
    "Cupper.Points", "Moisture", "Quakers", "altitude_low_meters",
    "altitude_high_meters", "altitude_mean_meters"
]
categorical_cols = [
    "Species", "Owner", "Country.of.Origin", "Mill", "ICO.Number", "Company", "Altitude",
    "Region", "Producer", "Bag.Weight", "In.Country.Partner", "Harvest.Year",
    "Grading.Date", "Owner.1", "Variety", "Processing.Method", "Color", "Expiration",
    "Certification.Body", "Certification.Address", "Certification.Contact", "unit_of_measurement"
]

# sanity check columns exist
missing_num = [c for c in numeric_cols if c not in df.columns]
missing_cat = [c for c in categorical_cols if c not in df.columns]
if missing_num or missing_cat:
    raise ValueError(f"Missing cols. numeric: {missing_num}, categorical: {missing_cat}")

# features and target
X = df[numeric_cols + categorical_cols].copy()
y = df[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# define pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ],
    remainder="drop",
    n_jobs=1
)

# fit preprocessor on training data only
preprocessor.fit(X_train)

# transform and convert to dense only for saving CSV (sparse->dense)
X_train_t = preprocessor.transform(X_train)
X_test_t  = preprocessor.transform(X_test)

def to_dense_df(X_t, feature_names, index):
    # X_t might be sparse
    if hasattr(X_t, "toarray"):
        arr = X_t.toarray()
    else:
        arr = X_t
    return pd.DataFrame(arr, columns=feature_names, index=index)

# build feature names
num_names = numeric_cols
ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
ohe_names = ohe.get_feature_names_out(categorical_cols).tolist()
feature_names = list(num_names) + list(ohe_names)

X_train_df = to_dense_df(X_train_t, feature_names, X_train.index)
X_test_df = to_dense_df(X_test_t, feature_names, X_test.index)

# save
X_train_df.to_csv("data/cleaned/X_train.csv", index=False)
X_test_df.to_csv("data/cleaned/X_test.csv", index=False)
y_train.to_csv("data/cleaned/y_train.csv", index=False)
y_test.to_csv("data/cleaned/y_test.csv", index=False)

# save preprocessor
joblib.dump(preprocessor, "artifacts/preprocessor.joblib")
print("Saved preprocessor to artifacts/preprocessor.joblib")
print("X_train shape:", X_train_df.shape)