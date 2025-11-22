"""
preprocess.py: 
- Reads preprocessed CSV.
- Verifies required columns.
- Splits into train/test.
- Fits and applies numeric + categorical pipelines.
- Saves cleaned train/test CSVs.
- Saves artifacts/preprocessor.joblib.

Citation:
OpenAI. (2025). ChatGPT (Version 5.1) [Large language model]. https://chat.openai.com  
Conversation with ChatGPT on November 20, 2025, used to generate some preprocessing and testing code snippets.
"""
import os, yaml, joblib, sklearn
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

"""
Cited block from chatGPT 5.1 at 10:10p on 11/20/25. 

Citation:
OpenAI. (2025). ChatGPT (Version 5.1) [Large language model]. https://chat.openai.com  
Conversation with ChatGPT on November 20, 2025, used to generate preprocessing code snippets.
"""

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

raw_path = config["data"]["local_path"]
preprocessed_path = config["data"]["preprocessed_path"]
target_col = config["data"]["target"]
test_size = config["train"]["test_size"]
random_state = config["train"]["random_state"]

"""
End cited block
"""

df = pd.read_csv(preprocessed_path)

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

# drop accidental index columns created by previous saves
df = df.loc[:, ~df.columns.str.contains(r'^Unnamed')]

# making sure columns exist
missing_num = [c for c in numeric_cols if c not in df.columns]
missing_cat = [c for c in categorical_cols if c not in df.columns]
if missing_num or missing_cat:
    raise ValueError(f"Missing cols. numeric: {missing_num}, categorical: {missing_cat}")

"""
The below block of code was derived from AIPI503 - Ed Lessons Day 4 Challenge
This course was taught by Dr. Daniel E. Davis, Ph.D.
"""

if target_col not in df.columns:
    raise ValueError(f"ERROR: Required target column '{target_col}' is missing from the dataset.")

X = df.drop(columns=[target_col])
y = df[target_col]

# Splitting data. 20% test, 80% train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

"""
End cited block
"""

# --- version-aware OneHotEncoder + categorical pipeline ---------------------
# due to running into sparse issues ChatGPT suggested this code snippet on 11/20/2025 at 8:53pm PST 

# defining numeric pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # missing numeric flled with median
    ("scaler", StandardScaler())   # standardize 
])

skl_ver = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
# build a OneHotEncoder compatible with this sklearn version
# the version check prevents the “sparse vs sparse_output” error
if skl_ver >= (1, 2):
    # sklearn 1.2+ uses sparse_output
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    # older sklearn uses sparse
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

# defining categorical pipeline 
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),   # missing categories filled w/most frequent 
    ("ohe", ohe)
])

# combine 2 pipelines into one obj that can full dataframes
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ],
    remainder="drop",
    n_jobs=1
)
# ----------------------------------END OF CITED BLOCK -----------------------------------------

# fit preprocessor on training data only
preprocessor.fit(X_train)

# transform train and test
X_train_t = preprocessor.transform(X_train)
X_test_t  = preprocessor.transform(X_test)


# Build feature names: pulls the encoded column names from 
# the fitted OneHotEncoder so the resulting DataFrames have readable column headers
num_features = numeric_cols.copy()   # numeric feature names are numeric_cols
cat_features = []
# access the fitted transformer objects safely
try:
    # ColumnTransformer stores the transformers under named_transformers_
    cat_tr = preprocessor.named_transformers_["cat"]
    # cat_tr is a Pipeline; get the OHE inside it
    if hasattr(cat_tr, "named_steps") and "ohe" in cat_tr.named_steps:
        fitted_ohe = cat_tr.named_steps["ohe"]
    else:
        # fallback: if the cat transformer itself is an OHE
        fitted_ohe = cat_tr
    # get feature names out (works on modern sklearn)
    try:
        cat_features = list(fitted_ohe.get_feature_names_out(categorical_cols))
    except Exception:
        # fallback: build generic names if method missing
        # prefix each category name with the column name
        cat_features = []
        # try transform a tiny sample to see number of output columns (not ideal)
        # but we can at least name them cat_0..cat_n
        n_cat_out = X_train_t.shape[1] - len(num_features)
        cat_features = [f"cat_{i}" for i in range(n_cat_out)]
except Exception as e:
    # If anything fails, fallback to generic names without breaking pipeline
    print("Warning building categorical feature names:", e)
    n_cat_out = X_train_t.shape[1] - len(num_features)
    cat_features = [f"cat_{i}" for i in range(n_cat_out)]

feature_names = num_features + cat_features

# converts the possibly sparse matrix to a regular Pandas DataFrame
def to_dense_df(X_t, feature_names, index):
    if hasattr(X_t, "toarray"):
        arr = X_t.toarray()
    else:
        arr = X_t
    return pd.DataFrame(arr, columns=feature_names, index=index)

X_train_df = to_dense_df(X_train_t, feature_names, X_train.index)
X_test_df  = to_dense_df(X_test_t,  feature_names, X_test.index)

# ensure directories exist
os.makedirs(os.path.dirname(config["paths"]["X_train"]), exist_ok=True)
os.makedirs(os.path.dirname(config["paths"]["X_test"]), exist_ok=True)
os.makedirs(os.path.dirname(config["paths"]["y_train"]), exist_ok=True)
os.makedirs(os.path.dirname(config["paths"]["y_test"]), exist_ok=True)
os.makedirs("artifacts", exist_ok=True)
# The above code snipet was generated by chatGPT 5.1 at 10:00p on 11/20/25.

# write 4 CSVs to the locations defined in config.yaml
X_train_df.to_csv(config["paths"]["X_train"], index=False)
X_test_df.to_csv(config["paths"]["X_test"], index=False)
y_train.to_csv(config["paths"]["y_train"], index=False)
y_test.to_csv(config["paths"]["y_test"], index=False)
# File locations generated by chatGPT 5.1 at 10:15p on 11/20/25.

# Save the fitted/trained preprocessing obj for later use (train.py & server)
# This was suggested by ChatGPT on 11/20/2025 around 8:20pm PST
joblib.dump(preprocessor, "artifacts/preprocessor.joblib")
print("Saved preprocessor to artifacts/preprocessor.joblib")
print("X_train shape:", X_train_df.shape)
print("X_test shape:", X_test_df.shape)
