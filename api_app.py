from fastapi import FastAPI, HTTPException
import json
import os

app = FastAPI(
    title="Credit Default Pipeline API",
    description="API to access data pipeline metrics and EDA details.",
    version="1.0.0"
)

LOG_FILE = "pipeline_metrics.json"

def get_latest_metrics():
    """Helper function to read the latest pipeline run metrics."""
    if not os.path.exists(LOG_FILE):
        raise HTTPException(status_code=404, detail="Metrics log not found. Pipeline hasn't run yet.")
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            if not lines:
                raise HTTPException(status_code=404, detail="Metrics log is empty.")
            latest_log = json.loads(lines[-1].strip())
            return latest_log.get("metrics", latest_log)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {str(e)}")

@app.get("/api/v1/health")
def health_check():
    """Application deployment and pipeline execution status."""
    try:
        metrics = get_latest_metrics()
        return {
            "status": "healthy",
            "deployment": "active",
            "last_pipeline_execution_status": metrics.get("execution_status", "unknown"),
            "last_execution_time_seconds": metrics.get("execution_time_seconds", 0)
        }
    except HTTPException:
        return {"status": "healthy", "deployment": "active", "pipeline_status": "waiting_for_first_run"}

@app.get("/api/v1/pipeline/flow")
def get_pipeline_flow():
    """Structural execution steps of the preprocessing and EDA pipeline."""
    return {
        "pipeline_steps": [
            {"step": 1, "name": "Data Ingestion", "description": "Download dataset from Kaggle and load into Pandas DataFrame."},
            {"step": 2, "name": "Data Preprocessing", "description": "Handle missing values via median imputation and scale numerical features using StandardScaler."},
            {"step": 3, "name": "Exploratory Data Analysis", "description": "Generate correlation matrix, bin continuous variables, and apply One-Hot Encoding."},
            {"step": 4, "name": "Feature Importance Extraction", "description": "Train a Random Forest classifier to extract Gini-based feature importances."},
            {"step": 5, "name": "DataOps & Automation", "description": "Schedule execution every 2 minutes and stream logs to centralized JSON."}
        ]
    }

@app.get("/api/v1/metrics/preprocessing")
def get_preprocessing_metrics():
    """Count of imputed missing values, record sizes, and scaling parameters."""
    metrics = get_latest_metrics()
    return {
        "records_processed": metrics.get("records_processed", 0),
        "missing_values_imputed": metrics.get("missing_values_imputed", 0),
        "scaling_method": "StandardScaler",
        "imputation_strategy": "median"
    }

@app.get("/api/v1/eda/feature-importance")
def get_feature_importance():
    """Top feature importance scores driving loan default predictions."""
    metrics = get_latest_metrics()
    return {
        "top_predictive_feature": metrics.get("top_feature_driver", "N/A"),
        "importance_score": metrics.get("top_feature_score", 0.0),
        "model_used": "RandomForestClassifier(n_estimators=50, max_depth=10)"
    }