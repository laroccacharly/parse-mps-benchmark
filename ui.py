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
    # Calculate Speedup Factor
    df['solve_time_speedup_factor'] = np.where(
        (df['direct_solve_time_seconds'].notna()) & 
        (df['parquet_solve_time_seconds'].notna()) & 
        (df['parquet_solve_time_seconds'] > 1e-9), # Avoid division by zero or near-zero parquet time
        df['direct_solve_time_seconds'] / df['parquet_solve_time_seconds'],
        np.nan # Assign NaN if either time is NaN or parquet time is too small
    )

    # Calculate objective match rate
    tolerance = 1e-6
    df['objectives_match'] = df['objective_diff_abs'] < tolerance

    return df

def display_histogram_with_stats(data_series, nbins, title, labels, stats_label, value_suffix="", value_format=".4f", caption=None):
    """Displays a Plotly histogram and summary statistics for a pandas Series."""
    if not data_series.empty:
        fig = px.histogram(data_series, nbins=nbins, title=title, labels=labels)
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

        # Calculate and display stats
        mean_val = data_series.mean()
        median_val = data_series.median()
        q1_val = data_series.quantile(0.25)
        q3_val = data_series.quantile(0.75)
        st.write(f"**Statistics ({stats_label}):**")
        st.write(f"- Mean: {mean_val:{value_format}}{value_suffix}")
        st.write(f"- Median: {median_val:{value_format}}{value_suffix}")
        st.write(f"- 25th Percentile: {q1_val:{value_format}}{value_suffix}")
        st.write(f"- 75th Percentile: {q3_val:{value_format}}{value_suffix}")
        if caption:
            st.caption(caption)
    else:
        st.write(f"No data available for {title}.")


# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("MPS parsing benchmark on MIPLIB")

st.markdown("## Research Question: Does converting MIP problems from the text-based MPS format to the binary Parquet format help speed up the HiGHS solver?")

# --- Display Key Takeaways ---
st.header("Key Takeaways")
takeaways_file = "key_takeaways.md"
if os.path.exists(takeaways_file):
    try:
        with open(takeaways_file, 'r') as f:
            takeaways_content = f.read()
        if takeaways_content:
            st.markdown(takeaways_content)
        else:
            st.info("The 'key_takeaways.md' file is empty. Add your key findings here.")
    except Exception as e:
        st.error(f"Error reading 'key_takeaways.md': {e}")
else:
    st.info("'key_takeaways.md' not found. Create this file in the project root to add key takeaways.")

# Load data
df_results = load_results(RESULTS_DIR)



if not df_results.empty:
    st.header("Results DataFrame")

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
        'direct_solve_time_seconds', 'parquet_solve_time_seconds', 'solve_time_diff_abs',
        'solve_time_speedup_factor',
        'parquet_load_create_time_seconds'
    ]].round(6)) # Round for display

    st.header("Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Solve Time Difference (Direct - Parquet)")
        # Absolute Difference
        plot_data_abs = df_results['solve_time_diff_abs'].dropna()
        display_histogram_with_stats(
            data_series=plot_data_abs,
            nbins=30,
            title="Absolute Difference in Solve Time",
            labels={'value': 'Absolute Time Difference (s)'},
            stats_label="Absolute Diff",
            value_suffix=" s",
            value_format=".4f"
        )

        # Parquet Load/Create Time (Moved here)
        st.subheader("Parquet Create Time")
        plot_data_load = df_results['parquet_load_create_time_seconds'].dropna()
        display_histogram_with_stats(
            data_series=plot_data_load,
            nbins=100, # Keep the increased bins here
            title="Distribution of Parquet Create Time",
            labels={'value': 'Time (s)'},
            stats_label="Create Time",
            value_suffix=" s",
            value_format=".4f",
            caption="Time taken to parse MPS, save to Parquet, and load back (or just load if exists)."
        )


    with col2:
        # Speedup Factor (Replaced Relative Gain)
        st.subheader("Solve Time Speedup Factor (Parquet vs Direct)")
        plot_data_speedup = df_results['solve_time_speedup_factor'].dropna()
        display_histogram_with_stats(
            data_series=plot_data_speedup,
            nbins=100,
            title="Solve Time Speedup Factor (Direct / Parquet)",
            labels={'value': 'Speedup Factor (x)'},
            stats_label="Speedup Factor",
            value_suffix="x",
            value_format=".2f",
            caption="Factor > 1 indicates Parquet solve was faster than direct MPS solve. Factor < 1 indicates slower."
        )


else:
    st.info("Load results data to view visualizations.") 