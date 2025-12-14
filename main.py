import os
from typing import Any

import pandas as pd
import mlflow
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "data", "datasets_iris.csv")
MODEL_NAME = "iris-tree-classifier"
MLFLOW_URI = os.getenv("MLFLOW_URI")


app = FastAPI(title="Iris API")

try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    experiment_name = "iris-pika-experiment"
    artifact_location = "gs://mlops-our-vigil-473719-c2/remote_repo"
    registered_model_name = "pikas-classifier"

    mlflow.set_tracking_uri(MLFLOW_URI)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        mlflow.set_experiment(experiment_name)
    else:
        mlflow.set_experiment(experiment_name)
        
except Exception as e:
    print(f"MLflow setup error: {e}")
    pass


class Flower(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float



def get_data() -> dict:
    if os.path.exists(DATA_CSV):
        return {"status": "exists", "path": DATA_CSV}
    return {"status": "error", "message": "Data file not found"}


@app.post("/train_model")
def train_model() -> Any:
    with mlflow.start_run(run_name="train_model"):
        info = get_data()
        mlflow.log_param("endpoint", "/train_model")
        if info.get("status").startswith("error"):
            mlflow.log_param("error", info.get("message"))
            raise HTTPException(status_code=500, detail=info.get("message"))
        
        df = pd.read_csv(DATA_CSV)
        mlflow.log_param("dataset_rows", df.shape[0])
        mlflow.log_param("dataset_columns", df.shape[1])
        
        X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        y = df["species"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=55)

        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 55)
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])

        param_grid = {"criterion": ["gini", "entropy"], "max_depth": [None, 5, 10]}
        gs = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid, cv=3)
        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_param("param_grid_criterion", "gini,entropy")
        mlflow.log_param("param_grid_max_depth", "None,5,10")
        mlflow.log_param("cv_folds", 3)
        
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        acc = metrics.accuracy_score(y_test, best.predict(X_test))

        mlflow.log_metric("test_accuracy", float(acc))
        mlflow.log_param("best_criterion", str(gs.best_params_.get("criterion")))
        mlflow.log_param("best_max_depth", str(gs.best_params_.get("max_depth")))
        mlflow.log_metric("best_cv_score", float(gs.best_score_))

        # Log model to MLflow (stored in GCS via MLflow server)
        mlflow.sklearn.log_model(
            best,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_test.iloc[:5] if len(X_test) > 0 else X_test
        )
        
        mlflow.log_metric("model_registered", 1)

        return {"status": "trained", "test_accuracy": round(float(acc), 4), "best_params": gs.best_params_}


@app.post("/fetch_model")
def fetch_model():
    with mlflow.start_run(run_name="fetch_model"):
        mlflow.log_param("endpoint", "/fetch_model")
        mlflow.log_param("model_name", MODEL_NAME)
        
        client = mlflow.tracking.MlflowClient()
        try:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            mlflow.log_param("model_versions_found", len(versions))
        except Exception as e:
            versions = []
            mlflow.log_param("model_versions_error", str(e))
            raise HTTPException(status_code=500, detail=f"Failed to fetch model versions: {str(e)}")
        
        if not versions:
            mlflow.log_param("model_available", False)
            raise HTTPException(status_code=404, detail="No registered model versions found")
        
        # Pick latest version by version number
        newest = max(versions, key=lambda v: int(getattr(v, "version", 0)))
        mlflow.log_param("latest_version", str(newest.version))
        mlflow.log_param("latest_stage", newest.current_stage)
        
        # Load model directly from MLflow Registry (stored in GCS)
        try:
            model_uri = f"models:/{MODEL_NAME}/{newest.version}"
            model = mlflow.sklearn.load_model(model_uri)
            mlflow.log_param("model_loaded", True)
            mlflow.log_param("model_uri", model_uri)
        except Exception as e:
            mlflow.log_param("model_load_error", str(e))
            raise HTTPException(status_code=500, detail=f"Failed to load model from registry: {str(e)}")

        return {"status": "ok", "model_name": MODEL_NAME, "model_version": str(newest.version), "model_uri": model_uri}


@app.post("/predict")
def predict(flower: Flower):
    with mlflow.start_run(run_name="predict"):
        mlflow.log_param("endpoint", "/predict")
        
        # Load latest model from MLflow Registry (stored in GCS)
        client = mlflow.tracking.MlflowClient()
        try:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if not versions:
                mlflow.log_param("model_available", False)
                raise HTTPException(status_code=404, detail="No registered model found")
            
            # Pick latest version
            newest = max(versions, key=lambda v: int(getattr(v, "version", 0)))
            model_uri = f"models:/{MODEL_NAME}/{newest.version}"
            model = mlflow.sklearn.load_model(model_uri)
            mlflow.log_param("model_version", str(newest.version))
            mlflow.log_param("model_uri", model_uri)
        except HTTPException:
            raise
        except Exception as e:
            mlflow.log_param("model_load_error", str(e))
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
        # Log input features
        data = flower.dict() if hasattr(flower, "dict") else flower.__dict__
        mlflow.log_param("sepal_length", float(data["sepal_length"]))
        mlflow.log_param("sepal_width", float(data["sepal_width"]))
        mlflow.log_param("petal_length", float(data["petal_length"]))
        mlflow.log_param("petal_width", float(data["petal_width"]))
        
        # Make prediction
        df = pd.DataFrame([data])
        df = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        pred = model.predict(df)[0]
        mlflow.log_param("prediction", str(pred))
        
        return {"prediction": str(pred)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)