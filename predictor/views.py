from django.shortcuts import render
import joblib
import numpy as np

# Load models once at server start
logistic_model = joblib.load("predictor/ml_models/logistic_model.pkl")
tree_model = joblib.load("predictor/ml_models/decision_tree_model.pkl")
feature_columns = joblib.load("predictor/ml_models/feature_columns.pkl")  # list of column names
NUM_FEATURES = len(feature_columns)  # e.g., 30

def home(request):
    prediction = None
    error = None

    if request.method == "POST":
        features_input = request.POST.get("features", "").strip()

        if features_input:  # Only validate if user typed something
            try:
                # Convert comma-separated string to float list
                features_list = [float(x.strip()) for x in features_input.split(",") if x.strip()]

                # Check if user entered fewer than required
                if len(features_list) < NUM_FEATURES:
                    error = f"You entered {len(features_list)} features, but the model needs {NUM_FEATURES} features."
                else:
                    # Trim extra features if user entered more
                    if len(features_list) > NUM_FEATURES:
                        features_list = features_list[:NUM_FEATURES]

                    # Convert to 2D array for sklearn
                    X_input = np.array([features_list])

                    # Predict
                    logistic_pred = logistic_model.predict(X_input)[0]
                    tree_pred = tree_model.predict(X_input)[0]

                    # Convert B/M to Benign/Malignant
                    logistic_pred = "Benign" if logistic_pred == "B" else "Malignant"
                    tree_pred = "Benign" if tree_pred == "B" else "Malignant"

                    prediction = {
                        "logistic": logistic_pred,
                        "tree": tree_pred
                    }

            except ValueError:
                error = "Please enter valid numbers separated by commas."

    return render(request, "index.html", {"prediction": prediction, "error": error})