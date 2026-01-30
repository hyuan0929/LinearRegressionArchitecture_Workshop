import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1) Define independent and dependent variables
independent_variable = "Number_Beds"
dependent_variable = "Price"

data = df[[independent_variable, dependent_variable]].copy()

# 2) Handle missing values
data[independent_variable] = pd.to_numeric(
    data[independent_variable], errors="coerce"
)
data[dependent_variable] = pd.to_numeric(
    data[dependent_variable], errors="coerce"
)

data = data.dropna()

# remove invalid values
data = data[
    (data[independent_variable] > 0) &
    (data[dependent_variable] > 0)
]

# 3) Split into X and y
X = data[[independent_variable]].values
y = data[dependent_variable].values

# 4) Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) Normalize / standardize feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Cleaned rows:", len(data))
print("Train shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)

# Import processed data into processed folder
import os

# Create processed directory if it does not exist
processed_dir = "data/processed"
os.makedirs(processed_dir, exist_ok=True)

# Save preprocessed data
processed_file_path = os.path.join(processed_dir, "housing_processed.csv")
data.to_csv(processed_file_path, index=False)

print(f"Preprocessed data saved to: {processed_file_path}")