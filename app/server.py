"""
app/server.py: 

Simple FastAPI server for Module 3 pipeline.
Endpoints:
  GET  /health       -> basic health check
  POST /predict      -> accepts JSON {"rows": [[...], [...]]} or single {"row":[...]}
                      returns predictions (uses artifacts/model.joblib and artifacts/preprocessor.joblib if present)
Behaviour:
  - If model or preprocessor are missing, server returns a sensible dummy prediction
  - Loads model & preprocessor once at startup

CITATION: 
- OpenAI. (2025). ChatGPT (Version 5.1) [Large language model]. https://chat.openai.com  
Conversation with ChatGPT on November 21, 2025, used to help generate most of FastAPI server code 
for serving the trained model and preprocessing pipeline (app/server.py).

"""

from fastapi import FastAPI, HTTPException   # api framework and exception handling imports 
from pydantic import BaseModel   # import to define and validate JSON request body
import joblib   # to load joblib object 
import numpy as np
import os, yaml, math
from typing import List, Dict, Any, Optional   # import for type hints 
from fastapi import Body
import pandas as pd

# point to config.yaml so server uses same paths as scripts
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
else:
    cfg = {}

# read artifacts paths from config 
PREPROCESSOR_PATH = cfg.get("artifacts", {}).get("preprocessor", "artifacts/preprocessor.joblib")
MODEL_PATH = cfg.get("artifacts", {}).get("model", "artifacts/model.joblib")
EXPECTED_COLS = cfg.get("data", {}).get("input_columns", None)
if EXPECTED_COLS is None:
    # is not present EXPECTED_COLS becomes empty
    EXPECTED_COLS = []

# create app object 
app = FastAPI(title="Coffee Quality - Prediction API")

# Dummy input model: either one row or multiple rows
class SingleRow(BaseModel):
    row: List[float]

class MultiRow(BaseModel):
    rows: List[List[float]]

# defines JSON body format expected by /predict_named
# expected body: { "rows": [ {"Aroma": 7.5, "Flavor": 6.0}, {"Aroma":5.0, "Flavor":7.0} ] }
class NamedRows(BaseModel):
    rows: List[Dict[str, Any]]


# loaded artifacts global vars (placeholders for now)
MODEL = None
PREPROCESSOR = None

# artifact loading function 
def load_artifacts():
    global MODEL, PREPROCESSOR
    # check if preprocessor exists 
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            # if preprocessor exists, its obj is loaded 
            PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
            app.state.preprocessor_loaded = True
        except Exception as e:
            PREPROCESSOR = None
            app.state.preprocessor_error = str(e)
    else:
        PREPROCESSOR = None
        app.state.preprocessor_loaded = False

     # check if model exists 
    if os.path.exists(MODEL_PATH):
        try:
            # if model exists, its obj is loaded 
            MODEL = joblib.load(MODEL_PATH)
            app.state.model_loaded = True
        except Exception as e:
            MODEL = None
            app.state.model_error = str(e)
    else:
        MODEL = None
        app.state.model_loaded = False

def build_rows_from_named(named_rows, expected_cols):
    """Turn list of dicts or a single dict into a 2-D array in the expected order."""
    import numpy as np
    if isinstance(named_rows, dict):
        named_rows = [named_rows]
    rows = []
    for r in named_rows:
        rows.append([r.get(col, np.nan) for col in expected_cols])
    return np.array(rows, dtype=float)

@app.post("/predict_named")
def predict_named(payload: NamedRows):
    # convert incoming list-of-dicts into DataFrame so that ColumnTransformer can accept and use it 
    try:
        df = pd.DataFrame(payload.rows)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid payload: rows must be a list of dicts")

    # If EXPECTED_COLS is set, reindex DF to match same column order & turn missing columns into NaN
    if EXPECTED_COLS:
        df = df.reindex(columns=EXPECTED_COLS)
    else:
        # if no config list, try to infer from preprocessor if possible
        try:
            cols = getattr(PREPROCESSOR, "feature_names_in_", None)
            if cols is not None:
                df = df.reindex(columns=list(cols))
        except Exception:
            pass

    # ensure dataframe rows are numeric for numeric columns when applicable — let preprocessor handle missing
    try:
        preds = _predict_with_artifacts(df)
        return {"predictions": preds}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid payload or transform error: {e}")

# startup event: when the app starts FastAPI is told to load artifacts
@app.on_event("startup")
def startup_event():
    load_artifacts()

# Health endpoint: checks if server is running and artifacts are loaded 
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bool(MODEL is not None),
        "preprocessor_loaded": bool(PREPROCESSOR is not None)
    }

# prediction helper that does ML work; supports both DataFrame input and raw numpy arrays
def _predict_with_artifacts(X):
    """
    Accepts X: either pandas.DataFrame (preferred) or 2D array-like (n_rows x n_features)
    Returns a JSON-safe Python list: floats where finite, None for NaN/Inf/unconvertible.
    """
     # If X is DataFrame, pass directly to PREPROCESSOR.transform 
    if isinstance(X, pd.DataFrame):
        X_in = X
    else:
        # convert to numpy array
        X_in = np.array(X, dtype=float)

    # if preprocessor is loaded, transforms X_in
    if PREPROCESSOR is not None:
        try:
            # ColumnTransformer supports DataFrame with column names and ndarray without names
            X_proc = PREPROCESSOR.transform(X_in)
        except Exception as e:
            raise RuntimeError(f"Preprocessor transform failed: {e}")
    else:
        # no preprocessor; assume X already numeric to avoid crash
        X_proc = np.array(X_in, dtype=float)

    # if model is loaded, calls predict () for real predictions 
    if MODEL is not None:
        preds = MODEL.predict(X_proc)
    else:
        # if not model loaded, return dummy: mean of row 
        preds = np.mean(X_proc, axis=1)
    # coerce to python list and make JSON-safe:
    safe_preds = []
    # convert to 1-D python list
    try:
        arr = np.asarray(preds).ravel().tolist()
    except Exception:
        # fallback: try to coerce single value
        arr = [preds]

    for p in arr:
        try:
            v = float(p)
            if not math.isfinite(v):
                # NaN / Inf -> null in JSON
                safe_preds.append(None)
            else:
                safe_preds.append(v)
        except Exception:
            # not convertible -> null
            safe_preds.append(None)

    return safe_preds
    # return preds.tolist()

# Predict endpoint: converts JSON rows into Numpy array and returns predictions as JSON list 
# good for batch predictions
@app.post("/predict")
def predict_multi(payload: MultiRow):
    try:
        X = np.array(payload.rows, dtype=float)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid input rows; must be numeric list of lists.")
    try:
        preds = _predict_with_artifacts(X)
        return {"predictions": preds}
    # if input is bad sends a proper HTTP error and code 
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

# predict row endpoint: converts a single JSON row into Numpy array and returns predictions as JSON list 
# good for one time testing such as via Postman
@app.post("/predict_row")
def predict_single(payload: SingleRow):
    try:
        X = np.array([payload.row], dtype=float)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid input row; must be numeric list.")
    try:
        preds = _predict_with_artifacts(X)
        return {"predictions": preds}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))