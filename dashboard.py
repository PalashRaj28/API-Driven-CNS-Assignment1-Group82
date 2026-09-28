import streamlit as st
import pandas as pd
import json
import time
import os

st.set_page_config(page_title="DataOps Pipeline Dashboard", layout="wide")

METRICS_LOG_FILE = "pipeline_metrics.json"

def load_metrics():
    try:
        with open(METRICS_LOG_FILE, "r") as f:
            # Read all lines and parse JSON for each line
            lines = f.readlines()
            metrics_list = [json.loads(line.strip())['metrics'] for line in lines if line.strip()]
            return metrics_list
    except (FileNotFoundError, json.JSONDecodeError):
        return []

st.title("DataOps Pipeline Monitoring Dashboard")
st.markdown("Visualizing real-time pipeline run status, record counts, and error metrics updated every 2 minutes.")

placeholder = st.empty()

while True:
    metrics_data = load_metrics()
    
    with placeholder.container():
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            latest = df.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Latest Status", str(latest.get('execution_status', 'N/A')).upper())
            col2.metric("Records Processed", f"{latest.get('records_processed', 0):,}")
            col3.metric("Missing Values Imputed", f"{latest.get('missing_values_imputed', 0):,}")
            
            # Count how many times 'failed' appears in the execution_status
            error_count = (df['execution_status'] == 'failed').sum() if 'execution_status' in df.columns else 0
            col4.metric("Total Errors", int(error_count))
            
            st.subheader("Recent Pipeline Runs")
            st.dataframe(df.tail(10), use_container_width=True)
            
            if 'execution_status' in df.columns:
                st.subheader("Execution History (Status)")
                status_counts = df['execution_status'].value_counts()
                st.bar_chart(status_counts)
        else:
            st.info("Waiting for the first pipeline execution to generate logs...")
            
    time.sleep(10) # Refresh dashboard UI every 10 seconds