"""}
MLflow Training Script for HistGradientBoosting Model
Implements MLOps best practices:
- Experiment tracking
- Model versioning
- Hyperparameter logging
- Metrics tracking
- Model registry
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import json
import os
from datetime import datetime

# Set MLflow tracking URI (can be configured to use remote server)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("histgradient-boosting-anomaly-detection")

def generate_data(n_samples=1000, n_features=20, n_informative=15, random_state=42):
    """
    Generate synthetic classification data for anomaly detection
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        random_state=random_state
    )
    return X, y

def train_model(params):
    """
    Train HistGradientBoosting model with MLflow tracking
    """
    with mlflow.start_run(run_name=f"hgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params(params)
        
        # Generate data
        X, y = generate_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Log data info
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Train model
        model = HistGradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log classification report as artifact
        report = classification_report(y_test, y_pred, output_dict=True)
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact("classification_report.json")
        
        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        np.savetxt("confusion_matrix.txt", cm, fmt='%d')
        mlflow.log_artifact("confusion_matrix.txt")
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="HistGradientBoosting-AnomalyDetector",
            signature=mlflow.models.infer_signature(X_train, y_pred)
        )
        
        # Log tags
        mlflow.set_tags({
            "model_type": "HistGradientBoosting",
            "task": "anomaly_detection",
            "framework": "scikit-learn",
            "environment": "production"
        })
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
        return model, accuracy

if __name__ == "__main__":
    # Define hyperparameters
    params = {
        "max_iter": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_samples_leaf": 20,
        "random_state": 42
    }
    
    # Train model
    model, accuracy = train_model(params)
    print(f"\nModel training complete with accuracy: {accuracy:.4f}")
    print("\nView results in MLflow UI: mlflow ui --port 5000")
