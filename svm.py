import yaml
import numpy as np
import pandas as pd
from sklearn.svm import SVC

def run_svm_from_yaml(config_file="svm.yaml"):
    """
    Loads parameters from a YAML config file,
    trains an SVM classifier, and makes predictions.
    """
    # Load config
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract model parameters
    model_type = config["model"]["type"]
    kernel = config["model"]["kernel"]
    random_state = config["model"]["random_state"]

    # Extract training data
    X = np.array(config["data"]["X"])
    y = np.array(config["data"]["y"])

    # Train classifier
    if model_type == "SVC":
        classifier = SVC(kernel=kernel, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model: {model_type}")

    classifier.fit(X, y)

    # Prediction
    X_marks = np.array(config["predict"]["input"])
    prediction = classifier.predict(X_marks)

    print("✅ Prediction result:", prediction.tolist())
    return prediction.tolist()


if _name_ == "_main_":
    run_svm_from_yaml("svm.yaml")
