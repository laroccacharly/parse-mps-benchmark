from miplib_benchmark.instance import get_instance_names, get_instance_path
import highspy
import numpy as np
import time
from parse_mps import parse_mps

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
        lp_data = parse_mps(path)
        parse_time = time.time() - start_time_parse
        print(f"MPS parsing completed in {parse_time:.4f} seconds")
    except Exception as e:
        print(f"Error during MPS parsing: {e}")
        return

    # 3. Solve using parsed data with HiGHS
    print("\nSolving with HiGHS using parsed data...")
    start_time_parsed = time.time()
    parsed_highs_result = solve_with_parsed_data(lp_data)
    solve_time_parsed = time.time() - start_time_parsed
    print(f"Parsed data HiGHS solve time: {solve_time_parsed:.4f} seconds")

    # 4. Compare results
    print("\n--- Results Comparison ---")
    print(f"Direct HiGHS Objective:   {direct_highs_result['objective_function_value']:.8f}")
    print(f"Parsed HiGHS Objective: {parsed_highs_result.get('objective_value', 'N/A'):.8f}")
    
    if parsed_highs_result.get('objective_value') is not None:
        diff = abs(direct_highs_result['objective_function_value'] - parsed_highs_result['objective_value'])
        print(f"Difference:               {diff:.8f}")
        if diff < 1e-6:
            print("Objective values match! Parser seems correct.")
        else:
            print("Objective values DIFFER! Check the parser.")
    else:
        print("Could not compare objective values (Parsed solve failed?).")

    # Optional: Compare status
    # print(f"Direct HiGHS Status: {direct_highs_result.model_status}") # Requires getting status correctly
    # print(f"Parsed HiGHS Status: {parsed_highs_result.get('status', 'N/A')}")

def solve_with_highs(path):
    model = highspy.Highs()
    model.setOptionValue("time_limit", 20.0) # Using float
    model.setOptionValue("solver", "simplex")
    model.readModel(str(path))
    model.run()
    # It's better to return the whole model object or necessary info 
    # highs_info = model.getInfo() 
    # return highs_info # Returning model allows getting more info later if needed
    # Let's return info directly for simplicity in this test
    info = model.getInfo()
    status = model.getModelStatus()
    obj_val = info.objective_function_value
    model.clear()
    # Return a dict for consistency
    return {'objective_function_value': obj_val, 'status': status, 'info': info}

def solve_with_parsed_data(lp_data):
    """Solve the LP using HiGHS, loading data from the parsed dictionary."""
    model = highspy.Highs()
    model.setOptionValue("time_limit", 20.0)
    model.setOptionValue("solver", "simplex")
    
    num_var = lp_data['n_vars']
    c = lp_data['c']
    lb, ub = lp_data['bounds']
    
    # Add variables with bounds and objective coefficients
    model.addVars(num_var, lb, ub)
    model.changeColsCost(num_var, np.arange(num_var), c)

    # Add constraints (handle equality and inequality separately)
    num_eq = lp_data['n_eq']
    num_ineq = lp_data['n_ineq']
    
    constraint_starts = []
    constraint_indices = []
    constraint_values = []
    constraint_lhs = []
    constraint_rhs = []

    current_start = 0

    # Process equality constraints (Ax = b -> b <= Ax <= b)
    if lp_data['A_eq'] is not None and num_eq > 0:
        A_eq = lp_data['A_eq']
        b_eq = lp_data['b_eq']
        for i in range(num_eq):
            row_indices = A_eq[i].nonzero()[0]
            row_values = A_eq[i, row_indices]
            constraint_starts.append(current_start)
            constraint_indices.extend(row_indices)
            constraint_values.extend(row_values)
            constraint_lhs.append(b_eq[i]) # Equality lower bound
            constraint_rhs.append(b_eq[i]) # Equality upper bound
            current_start += len(row_indices)

    # Process inequality constraints (Ax <= b -> -inf <= Ax <= b)
    if lp_data['A_ineq'] is not None and num_ineq > 0:
        A_ineq = lp_data['A_ineq']
        b_ineq = lp_data['b_ineq']
        for i in range(num_ineq):
            # A_ineq is currently a dense NumPy array from parse_mps
            row_indices = np.nonzero(A_ineq[i])[0] # Get indices of non-zero elements
            if row_indices.size == 0: # Skip if the row is all zeros
                continue 
            row_values = A_ineq[i, row_indices]
            
            constraint_starts.append(current_start)
            constraint_indices.extend(row_indices)
            constraint_values.extend(row_values)
            constraint_lhs.append(-highspy.kHighsInf) # Inequality lower bound (-inf)
            constraint_rhs.append(b_ineq[i])        # Inequality upper bound
            current_start += len(row_indices)

    # Add all constraints at once using addRows
    num_cons = num_eq + num_ineq
    if num_cons > 0:
        model.addRows(num_cons, np.array(constraint_lhs), np.array(constraint_rhs), 
                        len(constraint_indices), np.array(constraint_starts), 
                        np.array(constraint_indices, dtype=np.int32), np.array(constraint_values))

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
