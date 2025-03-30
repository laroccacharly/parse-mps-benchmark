import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import numpy as np

# Define the results directory
RESULTS_DIR = "data/runtime-benchmark"

@st.cache_data
def load_results(results_dir):
    """Loads all JSON results files from the specified directory into a pandas DataFrame."""
    all_results = []
    if not os.path.exists(results_dir):
        st.error(f"Results directory not found: {results_dir}")
        return pd.DataFrame()

    for filename in os.listdir(results_dir):
        if filename.endswith("_results.json"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # Handle potential None values before calculations
                    data['direct_objective_value'] = data.get('direct_objective_value')
                    data['parquet_objective_value'] = data.get('parquet_objective_value')
                    data['direct_solve_time_seconds'] = data.get('direct_solve_time_seconds')
                    data['parquet_solve_time_seconds'] = data.get('parquet_solve_time_seconds')
                    data['parquet_load_create_time_seconds'] = data.get('parquet_load_create_time_seconds')
                    all_results.append(data)
            except json.JSONDecodeError:
                st.warning(f"Could not decode JSON from file: {filename}")
            except Exception as e:
                st.warning(f"Error reading file {filename}: {e}")

    if not all_results:
        st.warning("No result files found or loaded.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    # --- Data Cleaning and Calculation ---
    # Replace potential None values from JSON loading with NaN for numeric operations
    numeric_cols = [
        'direct_objective_value', 'parquet_objective_value',
        'direct_solve_time_seconds', 'parquet_solve_time_seconds',
        'parquet_load_create_time_seconds'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') # 'coerce' turns errors into NaN

    # Calculate differences only where both values are not NaN
    df['objective_diff_abs'] = (df['direct_objective_value'] - df['parquet_objective_value']).abs()

    # Avoid division by zero or NaN/NaN for relative diff
    # Use np.isclose for near-zero checks
    df['objective_diff_rel'] = np.where(
        (df['direct_objective_value'].notna()) & (df['parquet_objective_value'].notna()) & (~np.isclose(df['direct_objective_value'], 0)),
        ((df['direct_objective_value'] - df['parquet_objective_value']) / df['direct_objective_value']).abs(),
        np.nan # Assign NaN if direct_objective is NaN, zero, or parquet_objective is NaN
    )

    df['solve_time_diff_abs'] = (df['direct_solve_time_seconds'] - df['parquet_solve_time_seconds'])

    df['solve_time_diff_rel'] = np.where(
        (df['direct_solve_time_seconds'].notna()) & (df['parquet_solve_time_seconds'].notna()) & (df['direct_solve_time_seconds'] > 1e-9), # Check for non-zero direct time
        ((df['direct_solve_time_seconds'] - df['parquet_solve_time_seconds']) / df['direct_solve_time_seconds']), # Keep sign for gain/loss
        np.nan
    )
    # Calculate relative gain (positive means parquet is faster)
    df['solve_time_gain_rel'] = df['solve_time_diff_rel'] # Positive means direct > parquet

    # Calculate objective match rate
    tolerance = 1e-6
    df['objectives_match'] = df['objective_diff_abs'] < tolerance

    return df

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("HiGHS MPS vs Parquet Benchmark Visualization")

# Load data
df_results = load_results(RESULTS_DIR)

if not df_results.empty:
    st.header("Benchmark Summary")

    # Objective Match Rate
    match_count = df_results['objectives_match'].sum()
    total_valid_comparisons = df_results['objective_diff_abs'].notna().sum()
    if total_valid_comparisons > 0:
        match_rate = (match_count / total_valid_comparisons) * 100
        st.metric(label="Objective Value Match Rate (<1e-6)", value=f"{match_rate:.2f}%",
                  help="Percentage of instances where direct and Parquet objectives are within 1e-6 absolute difference.")
    else:
        st.metric(label="Objective Value Match Rate (<1e-6)", value="N/A",
                  help="No valid objective comparisons available.")


    st.dataframe(df_results[[
        'instance_name',
        'direct_objective_value', 'parquet_objective_value', 'objective_diff_abs', 'objective_diff_rel', 'objectives_match',
        'direct_solve_time_seconds', 'parquet_solve_time_seconds', 'solve_time_diff_abs', 'solve_time_gain_rel',
        'parquet_load_create_time_seconds'
    ]].round(6)) # Round for display

    st.header("Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Solve Time Difference (Direct - Parquet)")
        # Filter out NaN values for plotting
        plot_data_abs = df_results['solve_time_diff_abs'].dropna()
        if not plot_data_abs.empty:
            fig_abs = px.histogram(plot_data_abs, nbins=30,
                                   title="Absolute Difference in Solve Time",
                                   labels={'value': 'Absolute Time Difference (s)'})
            fig_abs.update_layout(bargap=0.1)
            st.plotly_chart(fig_abs, use_container_width=True)
        else:
            st.write("No data available for absolute solve time difference histogram.")

        plot_data_rel = df_results['solve_time_gain_rel'].dropna() * 100 # Convert to percentage
        if not plot_data_rel.empty:
            fig_rel = px.histogram(plot_data_rel, nbins=30,
                                   title="Relative Gain in Solve Time (Parquet vs Direct)",
                                   labels={'value': 'Relative Time Gain (%)'})
            fig_rel.update_layout(bargap=0.1)
            st.plotly_chart(fig_rel, use_container_width=True)
            st.caption("Positive % indicates Parquet solve was faster than direct MPS solve.")
        else:
             st.write("No data available for relative solve time gain histogram.")


    with col2:
        st.subheader("Parquet Load/Create Time")
        plot_data_load = df_results['parquet_load_create_time_seconds'].dropna()
        if not plot_data_load.empty:
            fig_load = px.histogram(plot_data_load, nbins=30,
                                    title="Distribution of Parquet Load/Create Time",
                                     labels={'value': 'Time (s)'})
            fig_load.update_layout(bargap=0.1)
            st.plotly_chart(fig_load, use_container_width=True)
            st.caption("Time taken to parse MPS, save to Parquet, and load back (or just load if exists).")
        else:
            st.write("No data available for Parquet load/create time histogram.")


else:
    st.info("Load results data to view visualizations.") 