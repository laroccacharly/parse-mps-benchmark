import os
import time
import pandas as pd
import json
import numpy as np
from typing import Tuple, Optional
from scipy.sparse import coo_matrix
from lp_model import LpData  # Assuming LpData is defined here

def save_lp_to_parquet(lp_data: LpData, instance_name: str) -> Tuple[str, float]:
    """Saves LpData components to Parquet files inside a 'data' directory."""
    start_time = time.time()
    # Define the base data directory
    base_data_dir = "data"
    # Create the instance-specific directory path
    output_dir = os.path.join(base_data_dir, f"{instance_name}_parquet")
    # Create the base and instance-specific directories (exist_ok=True handles both)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving data to directory: {output_dir}")

    metadata = {
        'n_vars': lp_data.n_vars,
        # n_eq/n_ineq are derived properties in LpData, no need to store explicitly
        'obj_offset': lp_data.obj_offset
    }

    # Save vectors
    pd.DataFrame({'c': lp_data.c}).to_parquet(os.path.join(output_dir, 'c.parquet'))
    lb, ub = lp_data.bounds
    pd.DataFrame({'lb': lb, 'ub': ub}).to_parquet(os.path.join(output_dir, 'bounds.parquet'))

    if lp_data.b_eq.size > 0:
        pd.DataFrame({'b_eq': lp_data.b_eq}).to_parquet(os.path.join(output_dir, 'b_eq.parquet'))
    if lp_data.b_ineq.size > 0:
        pd.DataFrame({'b_ineq': lp_data.b_ineq}).to_parquet(os.path.join(output_dir, 'b_ineq.parquet'))

    # Save sparse matrix components (COO format) into single files
    def save_coo_matrix(row: Optional[np.ndarray], col: Optional[np.ndarray], data: Optional[np.ndarray], filename: str):
        # Ensure all components are present and non-empty before saving
        if row is not None and col is not None and data is not None and row.size > 0:
            # Assert they have the same size, just in case
            assert row.size == col.size == data.size, f"COO component size mismatch for {filename}"
            df = pd.DataFrame({'row': row, 'col': col, 'data': data})
            df.to_parquet(filename)
        # else: If any component is None or empty, don't save the file. Loader will handle missing files.

    save_coo_matrix(lp_data.A_eq_row, lp_data.A_eq_col, lp_data.A_eq_data, os.path.join(output_dir, 'A_eq_coo.parquet'))
    save_coo_matrix(lp_data.A_ineq_row, lp_data.A_ineq_col, lp_data.A_ineq_data, os.path.join(output_dir, 'A_ineq_coo.parquet'))

    save_time = time.time() - start_time
    print(f"Finished saving to Parquet in {save_time:.4f} seconds")

    # Add save_time to metadata
    metadata['save_time_seconds'] = save_time

    # Save updated metadata
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    # Return the path relative to the workspace root
    return output_dir, save_time 