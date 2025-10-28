# ------------------------------
# SIMPLE LINEAR REGRESSION - DATA PREPROCESSING
# ------------------------------

# 1️⃣ IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2️⃣ IMPORT THE DATASET
dataset = pd.read_csv(r"C:\Users\CHAITANYA\Desktop\Data.csv")

# 3️⃣ DEFINE INDEPENDENT (X) AND DEPENDENT (y) VARIABLES
# X = all columns except last
X = dataset.iloc[:, :-1].values
# y = last column (auto-detects even if column count changes)
y = dataset.iloc[:, -1].values

# 4️⃣ HANDLE MISSING NUMERICAL VALUES
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # Replace missing values with column mean
imputer.fit(X[:, 1:3])                    # Assuming numerical columns are at index 1 & 2
X[:, 1:3] = imputer.transform(X[:, 1:3])

# 5️⃣ ENCODE CATEGORICAL DATA (for first column of X)
from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# 6️⃣ ENCODE DEPENDENT VARIABLE IF IT’S CATEGORICAL
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# 7️⃣ SPLIT DATA INTO TRAINING & TEST SETS
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 8️⃣ PRINT CONFIRMATION
print("✅ Data successfully loaded and processed!\n")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nFirst 5 rows of X_train:")
print(X_train[:5])

# ------------------------------
# OPTIONAL: FEATURE SCALING (Recommended for ML models)
# ------------------------------
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print("\n✅ Feature scaling completed.")
print("Scaled X_train (first 5 rows):")
print(X_train[:5])


