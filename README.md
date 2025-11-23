# Module3Project

## Model
We used a RandomForestRegression for the model. Test size is 20% of dataset. Model has accuracy of 94.7% with 100 estimators.

## Data
For this project, we are using data on coffee quality found here:
https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi

# Setup:
```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
# Testing/running scripts
To test preprocess.py: 
```
python scripts/preprocess.py
```
Confirm all output files exist by running: 
```
ls -l data/cleaned/X_train.csv data/cleaned/X_test.csv data/cleaned/y_train.csv data/cleaned/y_test.csv artifacts/preprocessor.joblib
```
We wrote a unit test script tests/test_preprocessor.py, to run it: 
```
pip install pytest
pytest -q
```

To run the server, do health check use sample predict payload: 
```
uvicorn app.server:app --reload --port 8000 
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/predict_named" \
  -H "Content-Type: application/json" \
  -d '{"rows":[ {"Aroma":7.5,"Flavor":6.0,"Number.of.Bags":1,"Category.One.Defects":0} ] }'
```

To train the model:
'''
python train.py
'''
Ensure artifacts/model.joblib was built

# Notes / Gotchas
- config.yaml may include data.input_columns — if present the server will require/expect those columns and reindex incoming payloads automatically. 
- The server will try to load artifacts/preprocessor.joblib and artifacts/model.joblib. If those are missing the server returns deterministic dummy predictions (development mode).

# References: 
We used ChatGPT (OpenAI GPT-5.1) to assist with code snippets. 
Portions of the preprocessing and most of server code were assisted by ChatGPT (OpenAI GPT-5.1). Authors verified and adapted the generated code. 
Authors fully understand what the code does and how to apply the knowledge in the future. 