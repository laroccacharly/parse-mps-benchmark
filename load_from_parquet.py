import time
import os
import pandas as pd
import numpy as np
import json
from typing import Optional, Tuple

from lp_model import LpData
from parse_mps import parse_mps
from save_to_parquet import save_lp_to_parquet


def load_lp_from_parquet(instance_name: str, mps_path: str) -> Tuple[LpData, float]:
    """
    Loads LP data components from Parquet files.
    If the Parquet files don't exist for the instance, it parses the MPS file,
    saves the data to Parquet, and then loads it.
    Returns the LpData object and the time taken for loading (or parsing+saving+loading).
    """
    start_time_total = time.time() # Start timing the whole process

    base_data_dir = "data"
    data_dir = os.path.join(base_data_dir, f"{instance_name}_parquet")
    print(f"Checking for Parquet data directory: {data_dir}")

    # Check if the directory and a key file (e.g., metadata.json) exist
    metadata_path = os.path.join(data_dir, 'metadata.json')
    if not os.path.isdir(data_dir) or not os.path.exists(metadata_path):
        print(f"Parquet data not found for '{instance_name}'. Parsing MPS and saving to Parquet...")

        # 1. Parse MPS
        print(f"Parsing MPS file: {mps_path}")
        start_parse_save = time.time()
        try:
            lp_data_parsed = parse_mps(mps_path)
            parse_time = time.time() - start_parse_save
            print(f"MPS parsing completed in {parse_time:.4f} seconds")
        except Exception as e:
            print(f"Error during MPS parsing for lazy loading: {e}")
            raise # Re-raise the exception

        # 2. Save to Parquet
        print("Saving parsed data to Parquet...")
        _, save_time = save_lp_to_parquet(lp_data_parsed, instance_name)
        # data_dir should now exist

        # No need to load separately here, we already have lp_data_parsed
        # The function's primary goal becomes ensuring the data is available
        # and returning it. We can return the parsed data directly.
        total_time = time.time() - start_time_total
        print(f"Finished parsing and saving in {total_time:.4f} seconds (Parse: {parse_time:.4f}s, Save: {save_time:.4f}s)")
        # Return the *parsed* data directly as it's equivalent to loaded data
        return lp_data_parsed, total_time

    # If parquet data exists, proceed with loading
    print(f"Parquet data found. Loading from directory: {data_dir}")
    start_load_time = time.time() # Time only the loading part

    # --- Loading logic remains mostly the same ---
    lp_data_dict = {}

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    lp_data_dict['n_vars'] = metadata['n_vars']
    lp_data_dict['obj_offset'] = metadata.get('obj_offset', 0.0)

    # Load vectors
    lp_data_dict['c'] = pd.read_parquet(os.path.join(data_dir, 'c.parquet'))['c'].to_numpy()
    bounds_df = pd.read_parquet(os.path.join(data_dir, 'bounds.parquet'))
    lp_data_dict['bounds'] = (bounds_df['lb'].to_numpy(), bounds_df['ub'].to_numpy())

    b_eq_path = os.path.join(data_dir, 'b_eq.parquet')
    if os.path.exists(b_eq_path):
        lp_data_dict['b_eq'] = pd.read_parquet(b_eq_path)['b_eq'].to_numpy()
    else:
         lp_data_dict['b_eq'] = np.array([])

    b_ineq_path = os.path.join(data_dir, 'b_ineq.parquet')
    if os.path.exists(b_ineq_path):
        lp_data_dict['b_ineq'] = pd.read_parquet(b_ineq_path)['b_ineq'].to_numpy()
    else:
        lp_data_dict['b_ineq'] = np.array([])

    # Load sparse matrix COO components from single files
    def load_coo_matrix_internal(filename: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
         if not os.path.exists(filename):
             return None, None, None
         df = pd.read_parquet(filename)
         if df.empty:
             return None, None, None
         if 'row' in df.columns and 'col' in df.columns and 'data' in df.columns:
            return df['row'].to_numpy(), df['col'].to_numpy(), df['data'].to_numpy()
         else:
             print(f"Warning: Parquet file {filename} missing expected columns ('row', 'col', 'data'). Returning None.")
             return None, None, None

    lp_data_dict['A_eq_row'], lp_data_dict['A_eq_col'], lp_data_dict['A_eq_data'] = load_coo_matrix_internal(os.path.join(data_dir, 'A_eq_coo.parquet'))
    lp_data_dict['A_ineq_row'], lp_data_dict['A_ineq_col'], lp_data_dict['A_ineq_data'] = load_coo_matrix_internal(os.path.join(data_dir, 'A_ineq_coo.parquet'))

    lp_data_dict['col_names'] = None

    try:
        lp_data = LpData(**lp_data_dict)
    except Exception as e:
        print(f"Error creating LpData model from loaded data: {e}")
        raise e

    load_time = time.time() - start_load_time
    total_time = time.time() - start_time_total # This will be slightly more than load_time due to checks
    print(f"Finished loading from Parquet in {load_time:.4f} seconds (Total function time: {total_time:.4f}s)")
    # Return the *loaded* data
    return lp_data, total_time # Return total time for consistency 