import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("data.csv")

# Drop unnecessary columns
df = df.drop(columns=["id", "Unnamed: 32"], errors='ignore')

# Features and target
X = df.drop(columns=["diagnosis"], errors='ignore')  # features
y = df["diagnosis"]  # target column

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
logistic = LogisticRegression(max_iter=5000)
tree = DecisionTreeClassifier()

logistic.fit(X_train, y_train)
tree.fit(X_train, y_train)

# Create folder if it doesn't exist
os.makedirs("predictor/ml_models", exist_ok=True)

# Save models
joblib.dump(logistic, "predictor/ml_models/logistic_model.pkl")
joblib.dump(tree, "predictor/ml_models/decision_tree_model.pkl")

# ✅ Save feature columns for flexible input
joblib.dump(list(X.columns), "predictor/ml_models/feature_columns.pkl")

print("Models trained and saved successfully!")
