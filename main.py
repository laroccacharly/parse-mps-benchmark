from miplib_benchmark.instance import get_instance_path, get_instances
import highspy
import numpy as np
import time
from scipy.sparse import coo_matrix
from typing import Dict, Any
from lp_model import LpData
from load_from_parquet import load_lp_from_parquet
import os
import json

# Define the results directory as a constant
RESULTS_DIR = "data/runtime-benchmark"

def get_instances_names_sorted(): 
    instances_df = get_instances() 
    instances_df = instances_df.sort("n_variables")
    instances = instances_df["instance_name"].to_list()
    instances = instances[100:200]
    return instances

def process_instance(instance_name: str):
    """Processes a single instance: benchmarks and saves results if not already present."""
    # Use the constant RESULTS_DIR defined at the module level
    result_filepath = os.path.join(RESULTS_DIR, f"{instance_name}_results.json")

    if os.path.exists(result_filepath):
        print(f"Results file already exists for {instance_name}, skipping.")
        print("-" * (25 + len(instance_name))) # Separator
        return # Skip this instance

    # If file doesn't exist, proceed with benchmarking
    path = get_instance_path(instance_name)
    print(f"Using MPS file: {path}")

    print("\nSolving from mps file...")
    start_time_direct = time.time()
    try:
        direct_highs_result = solve_from_mps(path)
        solve_time_direct = time.time() - start_time_direct
        print(f"Direct HiGHS solve time: {solve_time_direct:.4f} seconds")
        obj_direct = direct_highs_result['objective_function_value']
    except Exception as e:
        print(f"Error solving directly from MPS for {instance_name}: {e}")
        solve_time_direct = time.time() - start_time_direct
        obj_direct = None
        direct_highs_result = {'status': 'Error', 'info': str(e)}

    print("\nLoading data from Parquet (or creating if needed)...")
    # The second return value 'load_time' now represents total time for this step
    # (either just loading, or parsing+saving+loading)
    try:
        lp_data_parquet, combined_load_time = load_lp_from_parquet(instance_name, path)
        parquet_loaded = True
    except Exception as e:
        print(f"Error loading/creating Parquet for {instance_name}: {e}")
        combined_load_time = 0.0 # Or time taken until error
        lp_data_parquet = None
        parquet_loaded = False
        parquet_highs_result = {'status': 'Error', 'info': 'Parquet loading failed'}
        solve_time_parquet = 0.0
        obj_parquet = None

    if parquet_loaded:
        print("\nSolving with HiGHS using Parquet-loaded data...")
        start_time_parquet_solve = time.time()
        try:
            parquet_highs_result = solve_from_lp_data(lp_data_parquet)
            solve_time_parquet = time.time() - start_time_parquet_solve
            print(f"Parquet data HiGHS solve time: {solve_time_parquet:.4f} seconds")
            obj_parquet = parquet_highs_result.get('objective_value')
        except Exception as e:
            print(f"Error solving from Parquet data for {instance_name}: {e}")
            solve_time_parquet = time.time() - start_time_parquet_solve
            obj_parquet = None
            parquet_highs_result = {'status': 'Error', 'info': str(e)}

    # 4. Compare and store results
    print("\n--- Results Comparison ---")
    print(f"Instance:                 {instance_name}")
    print(f"Direct HiGHS Objective:   {obj_direct if obj_direct is not None else 'N/A'} (Solve time: {solve_time_direct:.4f}s)")
    if parquet_loaded:
        print(f"Parquet HiGHS Objective:  {obj_parquet if obj_parquet is not None else 'N/A'} (Load/Create time: {combined_load_time:.4f}s, Solve time: {solve_time_parquet:.4f}s)")
    else:
        print("Parquet HiGHS Objective:  N/A (Load/Create failed)")

    if obj_direct is not None and obj_parquet is not None:
        diff_parquet = abs(obj_direct - obj_parquet)
        print(f"Difference (Direct vs Parquet):  {diff_parquet:.8f} {'(Match!)' if diff_parquet < 1e-6 else '(DIFFER!)'}")
    else:
        print("Difference (Direct vs Parquet):  N/A (Could not compare)")

    # 5. Save results to JSON (flattened structure)
    results_data = {
        "instance_name": instance_name,
        
        "direct_objective_value": obj_direct,
        "direct_solve_time_seconds": solve_time_direct,
        "parquet_load_create_time_seconds": combined_load_time if parquet_loaded else None,
        "parquet_objective_value": obj_parquet if parquet_loaded else None,
        "parquet_solve_time_seconds": solve_time_parquet if parquet_loaded else None,

    }

    try:
        with open(result_filepath, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"Results saved to {result_filepath}")
    except IOError as e:
        print(f"Error writing results file {result_filepath}: {e}")

    print("-" * (25 + len(instance_name))) # Separator

def main():
    instance_names = get_instances_names_sorted()
    total_instances = len(instance_names) 
    os.makedirs(RESULTS_DIR, exist_ok=True) 

    for i, instance_name in enumerate(instance_names):
        print(f"--- Processing instance {i+1}/{total_instances}: {instance_name} ---")

        process_instance(instance_name)

def solve_from_mps(path: str) -> Dict[str, Any]:
    model = highspy.Highs()
    model.setOptionValue("time_limit", 20.0) # Using float
    model.setOptionValue("solver", "simplex")
    model.setOptionValue("log_to_console", "false")
    model.readModel(str(path))
    model.run()
    info = model.getInfo()
    status = model.getModelStatus()
    obj_val = info.objective_function_value
    model.clear()
    # Return a dict for consistency
    return {'objective_function_value': obj_val, 'status': status, 'info': info}

def solve_from_lp_data(lp_data: LpData) -> Dict[str, Any]:
    """Solve the LP using HiGHS, loading data from the LpData model."""
    model = highspy.Highs()
    model.setOptionValue("time_limit", 20.0)
    model.setOptionValue("solver", "simplex")
    model.setOptionValue("log_to_console", "false")

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
    # Reconstruct A_eq from COO components if they exist
    if lp_data.A_eq_row is not None and lp_data.A_eq_col is not None and lp_data.A_eq_data is not None and num_eq > 0:
        A_eq_coo = coo_matrix((lp_data.A_eq_data, (lp_data.A_eq_row, lp_data.A_eq_col)), shape=(num_eq, num_var))
        A_eq_csr = A_eq_coo.tocsr() # Convert to CSR for HiGHS
        b_eq = lp_data.b_eq
        for i in range(num_eq):
            start = A_eq_csr.indptr[i]
            end = A_eq_csr.indptr[i+1]
            row_indices = A_eq_csr.indices[start:end]
            row_values = A_eq_csr.data[start:end]
            
            if len(row_indices) == 0: # Handle empty rows
                 constraint_starts.append(current_start)
                 constraint_lhs.append(b_eq[i])
                 constraint_rhs.append(b_eq[i])
                 continue

            constraint_starts.append(current_start)
            constraint_indices.extend(row_indices)
            constraint_values.extend(row_values)
            constraint_lhs.append(b_eq[i]) # Equality lower bound
            constraint_rhs.append(b_eq[i]) # Equality upper bound
            current_start += len(row_indices)
    elif num_eq > 0: # Handle case where b_eq exists but A_eq components are None/empty
        b_eq = lp_data.b_eq
        for i in range(num_eq):
            constraint_starts.append(current_start)
            constraint_lhs.append(b_eq[i])
            constraint_rhs.append(b_eq[i])

    # Process inequality constraints (Ax <= b -> -inf <= Ax <= b)
    # Reconstruct A_ineq from COO components if they exist
    if lp_data.A_ineq_row is not None and lp_data.A_ineq_col is not None and lp_data.A_ineq_data is not None and num_ineq > 0:
        A_ineq_coo = coo_matrix((lp_data.A_ineq_data, (lp_data.A_ineq_row, lp_data.A_ineq_col)), shape=(num_ineq, num_var))
        A_ineq_csr = A_ineq_coo.tocsr() # Convert to CSR for HiGHS
        b_ineq = lp_data.b_ineq
        for i in range(num_ineq):
            start = A_ineq_csr.indptr[i]
            end = A_ineq_csr.indptr[i+1]
            row_indices = A_ineq_csr.indices[start:end]
            row_values = A_ineq_csr.data[start:end]
            
            if len(row_indices) == 0: # Handle empty rows
                 constraint_starts.append(current_start)
                 constraint_lhs.append(-highspy.kHighsInf)
                 constraint_rhs.append(b_ineq[i])
                 continue
            
            constraint_starts.append(current_start)
            constraint_indices.extend(row_indices)
            constraint_values.extend(row_values)
            constraint_lhs.append(-highspy.kHighsInf) # Inequality lower bound (-inf)
            constraint_rhs.append(b_ineq[i])        # Inequality upper bound
            current_start += len(row_indices)
    elif num_ineq > 0: # Handle case where b_ineq exists but A_ineq components are None/empty
        b_ineq = lp_data.b_ineq
        for i in range(num_ineq):
            constraint_starts.append(current_start)
            constraint_lhs.append(-highspy.kHighsInf)
            constraint_rhs.append(b_ineq[i])

    # Add all constraints at once using addRows
    num_cons = len(constraint_starts) # Use length of starts to count actual constraints added
    if num_cons > 0:
        # Ensure indices and starts are numpy arrays of the correct type for HiGHS
        h_indices = np.array(constraint_indices, dtype=np.int32)
        h_starts = np.array(constraint_starts, dtype=np.int32)
        h_values = np.array(constraint_values, dtype=np.float64)
        h_lhs = np.array(constraint_lhs, dtype=np.float64)
        h_rhs = np.array(constraint_rhs, dtype=np.float64)
        
        model.addRows(num_cons, h_lhs, h_rhs, 
                        len(h_indices), h_starts, 
                        h_indices, h_values)

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

if __name__ == "__main__":
    main()
