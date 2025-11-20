import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/raw/raw_data.csv")

# Data Structure
print("Data Structure")
print("---------------")
print(f"Dimensions: {df.shape}")
print(f"Data Types:\n{df.dtypes}")
print(f"Missing Values:\n{df.isnull().sum()}")

"""
This data contains 1339 rows and 44 columns. 

"Number.of.Bags", "Category.One.Defects", and "Category.Two.Defects" are stored as integer data type.
"Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance", "Uniformity", "Clean.Cup", "Sweetness", "Cupper.Points", "Total.Cup.Points", "Moisture",
"Quakers", "altitude_low_meters", "altitude_high_meters", and "altitude_mean_meters" are stored as float data type. "Species", "Owner",
"Country.of.Origin", "Farm.Name", "Lot.Number", "Mill", "ICO.Number", "Company", "Altitude", "Region", "Producer", "Bag.Weight", "In.Country.Partner",
"Harvest.Year", "Grading.Date", "Owner.1", "Variety", "Processing.Method", "Color", "Expiration", "Certification.Body", "Certification.Address",
"Certification.Contact", and "unit_of_measurement" are stored as strings.
"""

# Owner is missing 7 entries. Owner should have impact on coffee quality, represented by Total.Cup.Points. Therefore, rows with empty entries can be dropped.
df.dropna(subset=["Owner"], inplace=True)

# Country.of.Origin (missing 1 value) will affect coffee quality, so instead of dropping this column, we will drop the row that is missing an entry.
df.dropna(subset=["Country.of.Origin"], inplace=True)

# Farm.Name is missing 359 entries. This has no impact of coffee quality, so drop this column.
df.drop(columns=["Farm.Name"], inplace=True)

# Lot.Number is missing 1063 entries. There are too many missing entries, so drop this column.
df.drop(columns=["Lot.Number"], inplace=True)

# Owner.1 is only missing 7 entries, drop rows where na.
df.dropna(subset=["Owner.1"], inplace=True)

# Quakers is only missing 1 entry, so throw out row where missing value
df.dropna(subset=["Quakers"], inplace=True)

"""
Allow the rest of this data to be preprocessed by the imputer in the data pipeline.
Columns that should not be filled by mean or most common removed here.
"""

# Data Structure After Manual Preprocessing
print("Data Structure After Manual Preprocessing")
print("---------------")
print(f"Dimensions: {df.shape}")
print(f"Missing Values:\n{df.isnull().sum()}")

df.to_csv("data/preprocessed/preprocessed_data.csv", index=False)