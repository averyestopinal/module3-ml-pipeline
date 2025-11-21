""" 
test_preprocessor.py: 
- unit testing of preprocessor.py

Citation:
OpenAI. (2025). ChatGPT (Version 5.1) [Large language model]. https://chat.openai.com  
Conversation with ChatGPT on November 20, 2025, used to generate some preprocessing and testing code snippets.
"""
import os
import pandas as pd
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# check artifact exists
def test_preprocessor_exists():
    assert os.path.exists("artifacts/preprocessor.joblib")

# CSVs are non-empty
def test_csvs_saved():
    for p in ["data/cleaned/X_train.csv","data/cleaned/X_test.csv"]:
        assert pd.read_csv(p).shape[0] > 0

# check train/test have same column count and no NaNs
def test_no_nans_and_matching_shapes():
    Xtr = pd.read_csv(config["paths"]["X_train"])
    Xte = pd.read_csv(config["paths"]["X_test"])
    assert Xtr.shape[1] == Xte.shape[1], "train/test have different number of columns"
    assert not Xtr.isnull().any().any(), "NaNs present in X_train"
    assert not Xte.isnull().any().any(), "NaNs present in X_test"

# quick asserts after saving
assert os.path.exists("artifacts/preprocessor.joblib")
Xtr = pd.read_csv(config["paths"]["X_train"])
Xte = pd.read_csv(config["paths"]["X_test"])
assert Xtr.shape[1] == Xte.shape[1], "train/test have different number of columns"
assert not Xtr.isnull().any().any(), "NaNs present in X_train"
assert not Xte.isnull().any().any(), "NaNs present in X_test"