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

To test preprocess.py: run `python scripts/preprocess.py`
Confirm all output files exist by running: 
`ls -l data/cleaned/X_train.csv data/cleaned/X_test.csv data/cleaned/y_train.csv data/cleaned/y_test.csv artifacts/preprocessor.joblib` 
We wrote a unit test script tests/test_preprocessor.py, to run it: 
```
pip install pytest

pytest -q
```