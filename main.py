from miplib_benchmark.instance import get_instance_names, get_instance_path
import highspy
import numpy as np
import time
from parse_mps import parse_mps
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
from scipy.sparse import csr_matrix, coo_matrix
from typing import Optional, Tuple, List, Dict, Any
from lp_model import LpData

def main():
    instance_names = get_instance_names()
    # Let's use a smaller instance for faster testing, e.g., '25fv47'
    # Adjust if needed, or stick with the first one
    instance_name = '25fv47' # instance_names[0]
    # Ensure the chosen instance exists
    if instance_name not in instance_names:
        print(f"Warning: Instance '{instance_name}' not found in miplib_benchmark. Using first instance: {instance_names[0]}")
        instance_name = instance_names[0]
        
    path = get_instance_path(instance_name)
    print(f"Using MPS file: {path}")

    # 1. Solve directly with HiGHS
    print("\nSolving directly with HiGHS...")
    start_time_direct = time.time()
    direct_highs_result = solve_with_highs(path)
    solve_time_direct = time.time() - start_time_direct
    print(f"Direct HiGHS solve time: {solve_time_direct:.4f} seconds")

    # 2. Parse the MPS file
    print("\nParsing MPS file...")
    start_time_parse = time.time()
    try:
        # Parse MPS now returns an LpData object directly
        lp_data_parsed = parse_mps(path)
        parse_time = time.time() - start_time_parse
        print(f"MPS parsing completed in {parse_time:.4f} seconds")
    except Exception as e:
        print(f"Error during MPS parsing or data validation: {e}")
        return

    # 3. Solve using parsed data with HiGHS
    print("\nSolving with HiGHS using initially parsed data...")
    start_time_parsed = time.time()
    parsed_highs_result = solve_with_parsed_data(lp_data_parsed)
    solve_time_parsed = time.time() - start_time_parsed
    print(f"Parsed data HiGHS solve time: {solve_time_parsed:.4f} seconds")
    
    # 3.5 Save parsed data to Parquet
    print("\nSaving parsed data to Parquet...")
    parquet_dir, save_time = save_lp_to_parquet(lp_data_parsed, instance_name)
    
    # 3.6 Load data from Parquet
    print("\nLoading data from Parquet...")
    lp_data_parquet, load_time = load_lp_from_parquet(instance_name)

    # 3.7 Solve using Parquet-loaded data with HiGHS
    print("\nSolving with HiGHS using Parquet-loaded data...")
    start_time_parquet = time.time()
    # Re-use the same solver function as it expects the same LpData structure
    parquet_highs_result = solve_with_parsed_data(lp_data_parquet)
    solve_time_parquet = time.time() - start_time_parquet
    print(f"Parquet data HiGHS solve time: {solve_time_parquet:.4f} seconds")

    # 4. Compare results
    print("\n--- Results Comparison ---")
    print(f"Instance:                 {instance_name}")
    print(f"Direct HiGHS Objective:   {direct_highs_result['objective_function_value']:.8f} (Solve time: {solve_time_direct:.4f}s)")
    print(f"Parsed HiGHS Objective:   {parsed_highs_result.get('objective_value', 'N/A'):.8f} (Parse time: {parse_time:.4f}s, Solve time: {solve_time_parsed:.4f}s)")
    print(f"Parquet HiGHS Objective:  {parquet_highs_result.get('objective_value', 'N/A'):.8f} (Save time: {save_time:.4f}s, Load time: {load_time:.4f}s, Solve time: {solve_time_parquet:.4f}s)")
    
    obj_direct = direct_highs_result['objective_function_value']
    obj_parsed = parsed_highs_result.get('objective_value')
    obj_parquet = parquet_highs_result.get('objective_value')

    if obj_parsed is not None:
        diff_parsed = abs(obj_direct - obj_parsed)
        print(f"Difference (Direct vs Parsed):   {diff_parsed:.8f} {'(Match!)' if diff_parsed < 1e-6 else '(DIFFER!)'}")
    else:
        print("Could not compare Direct vs Parsed (Parsed solve failed?).")

    if obj_parquet is not None:
        diff_parquet = abs(obj_direct - obj_parquet)
        print(f"Difference (Direct vs Parquet):  {diff_parquet:.8f} {'(Match!)' if diff_parquet < 1e-6 else '(DIFFER!)'}")
    else:
        print("Could not compare Direct vs Parquet (Parquet solve failed?).")

    # Optional: Compare status
    # print(f"Direct HiGHS Status: {direct_highs_result.model_status}") # Requires getting status correctly
    # print(f"Parsed HiGHS Status: {parsed_highs_result.get('status', 'N/A')}")
    # print(f"Parquet HiGHS Status: {parquet_highs_result.get('status', 'N/A')}")

def solve_with_highs(path: str) -> Dict[str, Any]:
    model = highspy.Highs()
    model.setOptionValue("time_limit", 20.0) # Using float
    model.setOptionValue("solver", "simplex")
    model.readModel(str(path))
    model.run()
    info = model.getInfo()
    status = model.getModelStatus()
    obj_val = info.objective_function_value
    model.clear()
    # Return a dict for consistency
    return {'objective_function_value': obj_val, 'status': status, 'info': info}

def solve_with_parsed_data(lp_data: LpData) -> Dict[str, Any]:
    """Solve the LP using HiGHS, loading data from the LpData model."""
    model = highspy.Highs()
    model.setOptionValue("time_limit", 20.0)
    model.setOptionValue("solver", "simplex")
    
    num_var = lp_data.n_vars
    c = lp_data.c
    lb, ub = lp_data.bounds
    
    # Add variables with bounds and objective coefficients
    model.addVars(num_var, lb, ub)
    model.changeColsCost(num_var, np.arange(num_var), c)

    # Add constraints
    num_eq = lp_data.n_eq
    num_ineq = lp_data.n_ineq
    
    constraint_starts = []
    constraint_indices = []
    constraint_values = []
    constraint_lhs = []
    constraint_rhs = []

    current_start = 0

    # Process equality constraints (Ax = b -> b <= Ax <= b)
    if lp_data.A_eq is not None and num_eq > 0:
        A_eq = lp_data.A_eq # Already csr_matrix
        b_eq = lp_data.b_eq
        for i in range(num_eq):
            start = A_eq.indptr[i]
            end = A_eq.indptr[i+1]
            row_indices = A_eq.indices[start:end]
            row_values = A_eq.data[start:end]
            
            if len(row_indices) == 0: # Skip empty rows if any
                # Note: HiGHS might require explicit handling or might ignore these
                # For simplicity, we add the constraint bounds but no coefficients
                 constraint_starts.append(current_start)
                 constraint_lhs.append(b_eq[i])
                 constraint_rhs.append(b_eq[i])
                 # current_start remains the same as no indices/values added
                 continue

            constraint_starts.append(current_start)
            constraint_indices.extend(row_indices)
            constraint_values.extend(row_values)
            constraint_lhs.append(b_eq[i]) # Equality lower bound
            constraint_rhs.append(b_eq[i]) # Equality upper bound
            current_start += len(row_indices)

    # Process inequality constraints (Ax <= b -> -inf <= Ax <= b)
    if lp_data.A_ineq is not None and num_ineq > 0:
        A_ineq = lp_data.A_ineq # Already csr_matrix
        b_ineq = lp_data.b_ineq
        for i in range(num_ineq):
            start = A_ineq.indptr[i]
            end = A_ineq.indptr[i+1]
            row_indices = A_ineq.indices[start:end]
            row_values = A_ineq.data[start:end]
            
            if len(row_indices) == 0: # Skip empty rows if any
                 # Add bounds for empty row: -inf <= 0 <= b_ineq[i]
                 constraint_starts.append(current_start)
                 constraint_lhs.append(-highspy.kHighsInf)
                 constraint_rhs.append(b_ineq[i])
                 # current_start remains the same
                 continue
            
            constraint_starts.append(current_start)
            constraint_indices.extend(row_indices)
            constraint_values.extend(row_values)
            constraint_lhs.append(-highspy.kHighsInf) # Inequality lower bound (-inf)
            constraint_rhs.append(b_ineq[i])        # Inequality upper bound
            current_start += len(row_indices)

    # Add all constraints at once using addRows
    num_cons = len(constraint_starts) # Use length of starts to count actual constraints added
    if num_cons > 0:
        model.addRows(num_cons, np.array(constraint_lhs), np.array(constraint_rhs), 
                        len(constraint_indices), np.array(constraint_starts), 
                        np.array(constraint_indices, dtype=np.int32), np.array(constraint_values))

    # Add objective offset if it exists in the data
    obj_offset = lp_data.obj_offset
    if obj_offset != 0.0:
      model.changeObjectiveOffset(obj_offset)

    # Solve the model
    model.run()
    
    # Get results
    info = model.getInfo()
    status = model.getModelStatus()
    obj_val = info.objective_function_value
    
    # Store results in a dictionary
    result_dict = {
        'success': status == highspy.HighsModelStatus.kOptimal,
        'status': status,
        'message': model.modelStatusToString(status),
        'objective_value': obj_val,
        'info': info
    }
    
    model.clear()
    return result_dict

def save_lp_to_parquet(lp_data: LpData, instance_name: str) -> Tuple[str, float]:
    """Saves LpData components to Parquet files."""
    start_time = time.time()
    output_dir = f"{instance_name}_parquet"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving data to directory: {output_dir}")

    metadata = {
        'n_vars': lp_data.n_vars,
        # Use properties for n_eq/n_ineq to be safe
        'n_eq': lp_data.n_eq, 
        'n_ineq': lp_data.n_ineq,
        'obj_offset': lp_data.obj_offset 
    }

    # Save metadata
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    # Save vectors
    pd.DataFrame({'c': lp_data.c}).to_parquet(os.path.join(output_dir, 'c.parquet'))
    lb, ub = lp_data.bounds
    pd.DataFrame({'lb': lb, 'ub': ub}).to_parquet(os.path.join(output_dir, 'bounds.parquet'))

    if lp_data.b_eq.size > 0:
        pd.DataFrame({'b_eq': lp_data.b_eq}).to_parquet(os.path.join(output_dir, 'b_eq.parquet'))
    if lp_data.b_ineq.size > 0:
        pd.DataFrame({'b_ineq': lp_data.b_ineq}).to_parquet(os.path.join(output_dir, 'b_ineq.parquet'))

    # Save sparse matrices (converting CSR to COO format for saving)
    def save_sparse_matrix(matrix: Optional[csr_matrix], filename: str, shape: Tuple[int, int]):
        if matrix is not None and matrix.nnz > 0:
            coo = matrix.tocoo()
            df = pd.DataFrame({'row': coo.row, 'col': coo.col, 'data': coo.data})
            df.to_parquet(filename)
            metadata[f'{os.path.basename(filename)}_shape'] = shape
        else: # Save empty structure if matrix is None or empty
            df = pd.DataFrame({'row': [], 'col': [], 'data': []})
            df.to_parquet(filename)

    # Matrices are already sparse or None
    save_sparse_matrix(lp_data.A_eq, os.path.join(output_dir, 'A_eq.parquet'), (metadata['n_eq'], metadata['n_vars']))
    save_sparse_matrix(lp_data.A_ineq, os.path.join(output_dir, 'A_ineq.parquet'), (metadata['n_ineq'], metadata['n_vars']))

    # Re-save metadata in case shape info was added for empty matrices
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    save_time = time.time() - start_time
    print(f"Finished saving to Parquet in {save_time:.4f} seconds")
    return output_dir, save_time


def load_lp_from_parquet(instance_name: str) -> Tuple[LpData, float]:
    """Loads LP data components from Parquet files and returns an LpData object."""
    start_time = time.time()
    data_dir = f"{instance_name}_parquet"
    print(f"Loading data from directory: {data_dir}")
    lp_data_dict = {}

    # Load metadata
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    lp_data_dict['n_vars'] = metadata['n_vars']
    # n_eq/n_ineq will be inferred by LpData model from matrix shapes
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

    # Load sparse matrices
    def load_sparse_matrix(filename: str, shape: Tuple[int, int]) -> Optional[csr_matrix]:
         if not os.path.exists(filename):
             print(f"Warning: File not found {filename}, returning None.")
             return None # Or appropriate empty matrix
         df = pd.read_parquet(filename)
         if df.empty:
             # Use shape info from metadata if saved
             actual_shape = metadata.get(f'{os.path.basename(filename)}_shape', shape)
             # Return an empty CSR matrix with the correct shape
             return csr_matrix(actual_shape)
         coo = coo_matrix((df['data'], (df['row'], df['col'])), shape=shape)
         return coo.tocsr() # Convert back to CSR for consistency with solver function

    # Use metadata shapes for loading
    n_eq = metadata['n_eq']
    n_ineq = metadata['n_ineq']
    n_vars = metadata['n_vars']
    
    lp_data_dict['A_eq'] = load_sparse_matrix(os.path.join(data_dir, 'A_eq.parquet'), (n_eq, n_vars))
    lp_data_dict['A_ineq'] = load_sparse_matrix(os.path.join(data_dir, 'A_ineq.parquet'), (n_ineq, n_vars))

    # Add col_names if needed (assuming not saved currently)
    lp_data_dict['col_names'] = None 

    # Create LpData instance
    try:
        lp_data = LpData(**lp_data_dict)
    except Exception as e:
        print(f"Error creating LpData model from loaded data: {e}")
        # Consider raising the exception or returning a specific error state
        raise e # Re-raise for now
        
    load_time = time.time() - start_time
    print(f"Finished loading from Parquet in {load_time:.4f} seconds")
    return lp_data, load_time

if __name__ == "__main__":
    main()
