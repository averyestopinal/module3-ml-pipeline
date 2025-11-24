# Module3Project

## Data
For this project, we are using data on coffee quality found here:
https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi

The cleaned coffee dataset is publicly hosted on Google Cloud Storage for reproducibility.
The preprocessing pipeline automatically downloads it via the data.url field in config.yaml.

Cleaned data in cloud:
https://storage.googleapis.com/coffee-quality-data/preprocessed_data.csv

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
python scripts/train.py
'''
Ensure artifacts/model.joblib was built

To run the UI app start the server and type in CLI: 
```
python app/frontend.py
Enter 3 when prompted:
  wandb: (1) Create a W&B account
  wandb: (2) Use an existing W&B account
  wandb: (3) Don't visualize my results
  My personal login is needed to sign in here to update to wandb website

```
Open link in browser

## Model
We used a RandomForestRegression for the model. Test size is 20% of dataset. Model has accuracy of 94.2% with 100 estimators.

W and B tracks model performance. Data can be found in wandb/run.../files/wandb-summary.json. Data is presented like this:
```
{
  "_timestamp":1.763876781125257e+09,
  "_wandb":{"runtime":2},
  "_runtime":2,
  "_step":0,
  "R2":0.9424069488737763,
  "RMSE":0.5528660703704987,
  "MAE":0.31615526315789416,
  "MAPE":0.39006294567905464
}
```
These perfomance metrics are also stored in artifacts.metrics.json like this:
```
{
    "R2": 0.9424069488737761,
    "RMSE": 0.5528660703704994,
    "MAE": 0.31615526315789455,
    "MAPE": 0.39006294567905514
}
```
The 94.2% R2 value shows very good fit and a cup score that correlates strongly with the other columns. The RMSE 0f 0.55 shows a small predicition error and therefore reinforces the model's high preformance.  The MAE of 0.314 also shows a small error to the actual cup points. MAPE shows average percentage error of 39% which shows medium accuracy. This could be due to the small size dataset the model was trained on.

# 🐳 Docker and Testing 
## Build the image 
```
# from the project root
docker build -t coffee-api:dev .
```

## Run the container 
```
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts":/app/artifacts \
  -v "$(pwd)/config.yaml":/app/config.yaml \
  -v "$(pwd)/data":/app/data \
  coffee-api:dev
```
Then open:
	•	Health check: http://127.0.0.1:8000/health￼
	•	Interactive docs: http://127.0.0.1:8000/docs￼

If artifacts are missing, the container automatically runs scripts/preprocess.py to generate them.

## Run tests inside the container 

To verify reproducibility of preprocessing and data pipeline:
```
docker run --rm -v "$(pwd)":/app -w /app coffee-api:dev python -m pytest -q
```
Expect output: 
```
...
3 passed in ~0.9s
```

## Docker-related notes: 
- Ports: container exposes 8000 (mapped to host port 8000)
- Artifacts (preprocessor.joblib, model.joblib) are mounted from the host for faster iteration


# Notes / Gotchas
- config.yaml may include data.input_columns — if present the server will require/expect those columns and reindex incoming payloads automatically. 
- The server will try to load artifacts/preprocessor.joblib and artifacts/model.joblib. If those are missing the server returns deterministic dummy predictions (development mode).

# References: 
We used ChatGPT (OpenAI GPT-5.1) to assist with code snippets. 
Portions of the preprocessing and most of server code were assisted by ChatGPT (OpenAI GPT-5.1). Authors verified and adapted the generated code. 
Authors fully understand what the code does and how to apply the knowledge in the future. 
