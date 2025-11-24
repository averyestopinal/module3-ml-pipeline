# app/frontend.py  
# Author: Eugenia Tate 
# Date: 11/23/2025

# CITATION: 
# ChatGPT was used to prototype robust JSON-sanitization and input-coercion logic when encountering serialization errors 
# and mixed user inputs (strings, noisy numeric text, pandas / numpy scalars). A recursive approach and conversion patterns 
# were suggested; we reviewed and thoroughly tested the code locally. See coerce_and_clamp_dict() and make_json_safe() below. 

# import necessary helpers 
import os
import yaml
import json
import math
import pandas as pd, numpy as np  # table handling 
import gradio as gr   # UI 
import requests   # to call API server 
from typing import Dict, Any, List

# point to config.yaml file to retrieve API URL 
CONFIG_PATH = os.path.join(os.getcwd(), "config.yaml")
# The above line was modified by ChatGPT 5.1 at 10:41a on 11/24/25 to work with Hugging Face
# if config exists - load it 
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
# if config does not exist - it falls back to being an empty dict
else:
    cfg = {}

# server endpoint UI will use for POST; if confid is missing fallback to predict_named
API_URL = cfg.get("api_url", {}).get("FastAPI", "http://127.0.0.1:8000/predict_named")

# reduced set of sensible columns exposed in UI to the end user
INPUT_COLS = [
    "Aroma", "Flavor", "Aftertaste", "Acidity",
    "Body", "Balance", "Sweetness", "Clean.Cup"
]
# help text for the end user explaining Clean Cup feature
CLEAN_CUP_HELP = (
    "Clean.Cup indicates the absence of off-flavors or defects (higher is better). "
    "Typically scored on the same sensory scale as other cup attributes."
)
# enforcing 0 to 10 possible values for input 
RANGES = {c: (0.0, 10.0) for c in INPUT_COLS}

# ------------------------------------ CITED BLOCK --------------------------------------------------------------------
# implemented using ChatGPT (conversation 2025-11-23) to help normalize free-form user input into numeric values within range
# convert user values to allowed 0 - 10 range to avoid errors/crashes: handles blanks, strings, noisy input by stripping chars 
# and sets None for missing / invalid entries (JSON's null)
def coerce_and_clamp_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    # out = {}
    out: Dict[str, Any] = {}
    # iterates over 8 input columns 
    for k in INPUT_COLS:
        v = row.get(k, "")
        # if a value user types is blank or string - converts it into np.nan
        # or if user types something like "7.5pts" it strips the letters and keeps the number 
        if v is None or (isinstance(v, str) and v.strip() == ""):
            # out[k] = np.nan
            out[k] = None
            continue
        # tries to convert to float 
        fv = None
        try:
            fv = float(v)
        except Exception:
            # try to strip out non-digit characters (e.g. "7.5pts" -> "7.5")
            try:
                cleaned = "".join(ch for ch in str(v) if (ch.isdigit() or ch in ".-"))
                fv = float(cleaned) if cleaned not in ("", ".", "-") else None
            except Exception:
                fv = None
        # if conversion failed -> None
        if fv is None or (isinstance(fv, float) and (math.isnan(fv) or math.isinf(fv))):
            out[k] = None
            continue
        # once we have a clean numeric - it is clamped to be within [0,10] range of valid inputs 
        # if user typed 13 it will be clmaped to 10
        # if user typed -2 it will become 0
        lo, hi = RANGES.get(k, (None, None))
        if lo is not None and hi is not None:
            fv = max(lo, min(hi, fv))
        out[k] = float(fv)
    # returns a clean dict to be sent to server 
    return out

# ChatGPT 5.1 used to prototype this recursive JSON-sanitizer 
# This function recursively walks nested containers (dicts, lists, tuples) and ensures any nested 
# structure (e.g. {"payload": [{"Aroma": np.nan}]}) becomes JSON-safe everywhere, not just the top level
def make_json_safe(obj):
    # dict
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    # numpy scalar -> python scalar
    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            return make_json_safe(obj.item())
    except Exception:
        pass
    # floats: map NaN/Inf -> None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    # ints, bool, str, None: ok
    if isinstance(obj, (int, bool, str)) or obj is None:
        return obj
    # fallback
    try:
        return str(obj)
    except Exception:
        return None
    

# ------------------------------------------ END CITED BLOCK ------------------------------------------------

# helper function that returns True if every value in a row is null or numeric 0, otherwise - False
def _row_is_all_null_or_zero(row: Dict[str, Any]) -> bool:
    for v in row.values():
        # missing/null -> keep scanning (counts as "no numeric input")
        if v is None:
            continue
        # numeric non-zero -> row is VALID
        if isinstance(v, (int, float)) and v != 0:
            return False
        # anything else (string, etc) is considered missing/invalid; continue 
        # but coerce_and_clamp_dict should have converted those to None or numeric
    return True

# sends JSON to server endpoint, returns a tuple (predictions list, raw resposnse/error)
def call_api_named(payload_rows: List[Dict[str, Any]]):
    # sanitize payload so it's JSON-serializable and uses `null` for missing
    safe_body = {"rows": make_json_safe(payload_rows)}
    try:
        payload_str = json.dumps(safe_body)
    except Exception as e:
        return None, f"Serialization error: {e}"
    # tries calling POST to get predictions using requests lib
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(API_URL, data=payload_str, headers=headers, timeout=10)   # timeout at 10 sec to avoid hanging 
        response.raise_for_status()
        # returns prediction list and full raw text response to be used within debug box on SUCCESS (200 OK) 
        return response.json().get("predictions", []), response.text
    except Exception as e:
        return None, f"API error: {e}"   # on error return None 

#prettifies prediction and debug JSON 
def predict_from_rows_of_dicts(rows_of_dicts: List[Dict[str, Any]]):
    payload_rows = [coerce_and_clamp_dict(row) for row in rows_of_dicts]
    # decide whether submission is allowed:
    # - if every submitted row is all-null-or-zero, refuse
    all_rows_invalid = all(_row_is_all_null_or_zero(r) for r in payload_rows)
    if all_rows_invalid:
        debug = {"payload": payload_rows, "response_raw": "skipped - all values missing or zero"}
        return "Please enter at least one numeric attribute (non-zero) before submitting.", json.dumps(debug, indent=2)
    # Otherwise proceed and call API (allowed if at least one row has a non-zero numeric)
    preds, raw = call_api_named(payload_rows)
    # building a debug dictionary containing both payload and raw server response
    debug = {"payload": payload_rows, "response_raw": raw}
    # if API fails - return empty prediction and debug JSON for debugging 
    if preds is None:
        return "", json.dumps(debug, indent=2)
    # prettifying predictions upon successful call to be user-friendly 
    prettified_pred = [f"Predicted Coffee Quality Points = {round(float(p), 1)}" for p in preds]   # rounding predictions to 1 decimal place (user friendly)
    #returns prettified prediction and debug JSON for debug box 
    return "\n".join(prettified_pred), json.dumps(debug, indent=2)


def predict_from_table(table):
    rows_of_dicts = table_to_list_of_dicts(table)
    return predict_from_rows_of_dicts(rows_of_dicts)


# ------------------------------------ CITED BLOCK -------------------------------------
# ChatGPT was used on 11/23/2025 to fix this function due to encountering errors to help deal 
# with 2 possible incoming formats: Dataframe and list of lists. 

# helper function puts input into proper expected by server format of list-of-dicts keyed by INPUT_COLS:
# [{"Aroma": 7.5, "Flavor": 8.0, ...}]; 
# fills missing columns with empty strings so coerce_and_clamp_dict() can convert them to np.nan
def table_to_list_of_dicts(table):
    # if table passed in is an instance of Dataframe obj - turn it into a dict 
    if isinstance(table, pd.DataFrame):
        df = table
        return [df.iloc[i].to_dict() for i in range(len(df))]
    # else - assume table is a list of lists and manually pair each element to corresponding column 
    rows = []
    for row in table:
        # ensure row has right length
        vals = list(row) + [""] * max(0, len(INPUT_COLS) - len(row))
        rows.append({col: vals[i] for i, col in enumerate(INPUT_COLS)})
    return rows
# ------------------------------- END CITED BLOCK -------------------------------------------


# -------------------------------- Gradio UI ------------------------------------------------------
with gr.Blocks(title="Coffee Quality Points Estimator") as demo:
    # inline HTML/CSS to style user instructions 
    gr.Markdown("<h1 style='text-align:center;color:#08306B'>Coffee Quality Points Estimator</h1>")
    gr.Markdown(
        "<div style='font-size:17px;font-weight:700;color:#2b6cb0'>"
        "Instructions: Fill the known sensory attributes (0–10). Leave unknowns blank and the model will "
        "attempt to infer missing values. Then click <b style='color:#ff6600'>Submit</b> to estimate the "
        "<b>Coffee Quality Points</b> (Total.Cup.Points). Higher scores mean better coffee quality.</div>"
    )

    with gr.Row():
        # presents 1 row by default with INPUT_COLS 
        df_input = gr.Dataframe(
            headers=INPUT_COLS,
            value=[["" for _ in INPUT_COLS]],    # list of lists to avoid validation errors encountered on testing 
            # ------------------------- ChatGPT 5.1 was used to fix the issues on 11/23/2025 ---------------------
            row_count=1,
            col_count=len(INPUT_COLS),
            interactive=True,
            label="Enter Known Columns (0–10 range; numeric values preferred)"
        )

    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")

    with gr.Row():
        # short prediction for the user 
        pred_out = gr.Textbox(label="Predicted Coffee Quality Points", lines=1, interactive=False)

    with gr.Row():
        # full debug info for developer 
        debug_out = gr.Textbox(label="Debug (payload + raw response)", lines=10, interactive=False)

    with gr.Row():
        gr.Markdown(f"<b>Note:</b> <i>{CLEAN_CUP_HELP}</i>")

    # When user clicks Submit, Gradio sends the contents of the table to table_to_list_of_dicts(). 
    # the content can either be a Dataframe or list of lists and the helper function can handle both
    # making the format consistent with FastAPI expectations
    def submit_table(table):
        rows_of_dicts = table_to_list_of_dicts(table)
        return predict_from_rows_of_dicts(rows_of_dicts)
    
    # fires up the actual prediction
    submit_btn.click(predict_from_table, inputs=[df_input], outputs=[pred_out, debug_out])

if __name__ == "__main__":
    # auto opens the demo in browser 
    demo.launch()