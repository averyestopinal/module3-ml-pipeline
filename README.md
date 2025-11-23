# Module3Project


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