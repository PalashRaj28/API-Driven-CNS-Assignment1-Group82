import os
import pandas as pd
import kagglehub
import logging
import time
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from apscheduler.schedulers.blocking import BlockingScheduler

# ==========================================
# Task 1.1: Environment Setup
# ==========================================
def setup_directories():
    """Create the required project repository with structured directories."""
    directories = ['./data', './src/preprocessing', './logs']
    for d in directories:
        os.makedirs(d, exist_ok=True)

def ingest_and_preprocess():
    setup_directories()
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("./logs/preprocessing.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting data ingestion from Kaggle...")
    
    path = kagglehub.dataset_download("brycecf/give-me-some-credit-dataset")
    file_path = os.path.join(path, "cs-training.csv")
    df = pd.read_csv(file_path)
    
    initial_rows, initial_cols = df.shape
    logging.info(f"Dataset loaded successfully. Shape: {initial_rows} rows, {initial_cols} columns.")
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    numeric_cols = ['DebtRatio', 'MonthlyIncome', 'RevolvingUtilizationOfUnsecuredLines']
    stats_df = pd.DataFrame({
        'Mean': df[numeric_cols].mean(),
        'Median': df[numeric_cols].median(),
        'Std_Dev': df[numeric_cols].std(),
        'Skewness': df[numeric_cols].skew()
    })
    logging.info(f"Summary Statistics calculated:\n{stats_df}")
    
    missing_income = df['MonthlyIncome'].isnull().sum()
    missing_dependents = df['NumberOfDependents'].isnull().sum()
    total_imputed = missing_income + missing_dependents
    
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())
    
    df['NumberOfDependents'] = df['NumberOfDependents'].astype(int)
    df['SeriousDlqin2yrs'] = df['SeriousDlqin2yrs'].astype(int)
    
    features_to_scale = df.columns.drop('SeriousDlqin2yrs')
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    processed_file_path = './data/processed_credit_data.csv'
    df.to_csv(processed_file_path, index=False)

    preprocessing_metadata = {
        "pipeline_step": "preprocessing",
        "initial_records": int(initial_rows),
        "missing_values_imputed": int(total_imputed),
        "imputation_strategy": "median",
        "status": "success"
    }
    
    return df, preprocessing_metadata


# ==========================================
# Task 2: Exploratory Data Analysis
# ==========================================
def perform_eda(df):
    logging.info("Starting EDA phase...")
    
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr(method='pearson')
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    plt.title('Pearson Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    df_engineered = df.copy()
    age_bins = [0, 25, 45, 65, 120]
    age_labels = ['Under_25', '25_to_45', '45_to_65', 'Over_65']
    df_engineered['AgeGroup'] = pd.cut(df_engineered['age'], bins=age_bins, labels=age_labels, right=False)
    
    income_bins = [0, 3000, 5000, 8000, 15000, np.inf]
    income_labels = ['Low', 'Lower_Middle', 'Middle', 'Upper_Middle', 'High']
    df_engineered['IncomeGroup'] = pd.cut(df_engineered['MonthlyIncome'], bins=income_bins, labels=income_labels, right=False)
    
    df_encoded = pd.get_dummies(df_engineered, columns=['AgeGroup', 'IncomeGroup'], drop_first=False)
    
    target_col = 'SeriousDlqin2yrs'
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    rf_model.fit(X, y)
    
    feat_imp_df = pd.DataFrame({
        'Feature': X.columns, 
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
    plt.title('Random Forest Feature Importance (Gini)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['DebtRatio'], bins=50, ax=axes[0], color='blue', kde=False)
    axes[0].set_yscale('log')
    axes[0].set_title('Distribution of DebtRatio (Log Scale)')
    
    sns.boxplot(x=df['RevolvingUtilizationOfUnsecuredLines'], ax=axes[1], color='orange')
    axes[1].set_xscale('log')
    axes[1].set_title('Boxplot: Revolving Utilization')
    
    default_rates = df_engineered.groupby('AgeGroup', observed=True)['SeriousDlqin2yrs'].mean().reset_index()
    default_rates['SeriousDlqin2yrs'] *= 100 
    sns.barplot(x='AgeGroup', y='SeriousDlqin2yrs', data=default_rates, palette='magma', ax=axes[2])
    axes[2].set_title('Default Rate (%) by Age Bracket')
    
    plt.tight_layout()
    plt.savefig('eda_plots.png')
    plt.close()
    
    eda_metadata = {
        "pipeline_step": "exploratory_data_analysis",
        "top_predictive_feature": str(feat_imp_df.iloc[0]['Feature']),
        "top_feature_importance_score": round(float(feat_imp_df.iloc[0]['Importance']), 4),
        "visualizations_generated": 3,
        "status": "success"
    }
    return df_encoded, eda_metadata


# ==========================================
# Task 3: Pipeline Scripting & Logging
# ==========================================
LOG_FILE = 'pipeline_metrics.json'

def log_pipeline_metrics(metrics_data):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "metrics": metrics_data
    }
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    logging.info(f"PIPELINE METRICS LOGGED: {json.dumps(metrics_data)}")

def run_pipeline():
    logging.info("--- STARTING AUTOMATED PIPELINE RUN ---")
    start_time = time.time()
    try:
        df, prep_metadata = ingest_and_preprocess()
        df_encoded, eda_metadata = perform_eda(df)
        
        execution_time = round(time.time() - start_time, 2)
        pipeline_metrics = {
            "execution_status": "success",
            "execution_time_seconds": execution_time,
            "records_processed": prep_metadata["initial_records"],
            "missing_values_imputed": prep_metadata["missing_values_imputed"],
            "top_feature_driver": eda_metadata["top_predictive_feature"],
            "top_feature_score": eda_metadata["top_feature_importance_score"]
        }
        
        log_pipeline_metrics(pipeline_metrics)
        logging.info("--- PIPELINE RUN COMPLETED SUCCESSFULLY ---")
        return pipeline_metrics
        
    except Exception as e:
        execution_time = round(time.time() - start_time, 2)
        error_metrics = {
            "execution_status": "failed",
            "execution_time_seconds": execution_time,
            "error_message": str(e)
        }
        log_pipeline_metrics(error_metrics)
        logging.error(f"--- PIPELINE RUN FAILED: {str(e)} ---")
        return error_metrics


if __name__ == "__main__":
    # Run it once immediately when the script starts
    test_run_metrics = run_pipeline()

    # Initialize the blocking scheduler
    scheduler = BlockingScheduler()

    # Add the pipeline job to run every 2 minutes
    scheduler.add_job(
        func=run_pipeline, 
        trigger="interval", 
        minutes=2,
        id='credit_default_pipeline_job',
        name='Run Data Ingestion and EDA Pipeline every 2 mins',
        replace_existing=True
    )

    # Start the scheduler
    logging.info("Scheduler started. Pipeline will execute every 2 minutes. Press Ctrl+C in the terminal to halt.")
    scheduler.start()