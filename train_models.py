from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
logistic = LogisticRegression(max_iter=5000)
tree = DecisionTreeClassifier()

logistic.fit(X_train, y_train)
tree.fit(X_train, y_train)

# Create folder if not exists
os.makedirs("predictor/ml_models", exist_ok=True)

# Save models
joblib.dump(logistic, "predictor/ml_models/logistic_model.pkl")
joblib.dump(tree, "predictor/ml_models/decision_tree_model.pkl")

print("✅ Models trained and saved successfully!")
