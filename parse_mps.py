import numpy as np
import time

def parse_mps(path):
    """Parse an MPS file into a format suitable for simplex solver."""
    parse_start_time = time.time()
    print(f"Starting MPS parsing...")
    
    # Initialize sections
    current_section = None
    row_names = []
    col_names = []
    objective_name = None
    constraints = {}
    objective = {}
    rhs_values = {}
    bounds = {}
    
    # Row types (N=objective, E=equality, L=less than, G=greater than)
    row_types = {}
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
                
            # Identify section headers
            if line in ['NAME', 'ROWS', 'COLUMNS', 'RHS', 'RANGES', 'BOUNDS', 'ENDATA']:
                current_section = line
                continue
                
            if current_section == 'ROWS':
                parts = line.split()
                row_type = parts[0]
                row_name = parts[1]
                row_names.append(row_name)
                row_types[row_name] = row_type
                
                if row_type == 'N':
                    objective_name = row_name
            
            elif current_section == 'COLUMNS':
                # Skip markers
                if "'MARKER'" in line:
                    continue
                
                parts = line.split()
                col_name = parts[0]
                
                if col_name not in col_names:
                    col_names.append(col_name)
                
                # Handle entries with multiple coefficients on a line
                i = 1
                while i < len(parts):
                    row_name = parts[i]
                    value = float(parts[i+1])
                    
                    if row_name == objective_name:
                        objective[col_name] = value
                    else:
                        if row_name not in constraints:
                            constraints[row_name] = {}
                        constraints[row_name][col_name] = value
                    
                    i += 2
            
            elif current_section == 'RHS':
                parts = line.split()
                # Skip RHS name
                rhs_name = parts[0]
                
                # Process each rhs entry
                i = 1
                while i < len(parts):
                    row_name = parts[i]
                    value = float(parts[i+1])
                    rhs_values[row_name] = value
                    i += 2
            
            elif current_section == 'BOUNDS':
                parts = line.split()
                bound_type = parts[0]
                # Skip bound name
                bound_name = parts[1]
                col_name = parts[2]
                value = float(parts[3]) if len(parts) > 3 else 0.0
                
                if col_name not in bounds:
                    # Default bounds [0, +inf]
                    bounds[col_name] = {"LO": 0.0, "UP": float('inf')}
                
                if bound_type == 'LO':
                    bounds[col_name]["LO"] = value
                elif bound_type == 'UP':
                    bounds[col_name]["UP"] = value
                elif bound_type == 'FX':
                    bounds[col_name]["LO"] = value
                    bounds[col_name]["UP"] = value
                elif bound_type == 'FR':
                    bounds[col_name]["LO"] = float('-inf')
                    bounds[col_name]["UP"] = float('inf')
                elif bound_type == 'MI':
                    bounds[col_name]["LO"] = float('-inf')
                elif bound_type == 'PL':
                    bounds[col_name]["UP"] = float('inf')
                elif bound_type == 'BV':
                    bounds[col_name]["LO"] = 0.0
                    bounds[col_name]["UP"] = 1.0
    
    print(f"Finished reading MPS sections in {time.time() - parse_start_time:.2f} seconds")
    
    # Set default bounds for variables without explicit bounds
    for col in col_names:
        if col not in bounds:
            bounds[col] = {"LO": 0.0, "UP": float('inf')}
    
    # Convert to matrix form (minimize c·x subject to Ax = b, lb ≤ x ≤ ub)
    matrix_start_time = time.time()
    print(f"Starting matrix conversion...")
    
    n_vars = len(col_names)
    n_constraints = len(row_names) - 1  # Excluding objective row
    
    # Create coefficient matrix A and RHS vector b
    A = np.zeros((n_constraints, n_vars))
    b = np.zeros(n_constraints)
    
    # Fill objective coefficients
    c = np.zeros(n_vars)
    for i, col in enumerate(col_names):
        c[i] = objective.get(col, 0.0)
    
    # Fill constraint matrix and RHS
    constraint_idx = 0
    constraint_types = []
    
    for row in row_names:
        if row == objective_name:
            continue
        
        row_type = row_types[row]
        constraint_types.append(row_type)
        
        for j, col in enumerate(col_names):
            if row in constraints and col in constraints[row]:
                A[constraint_idx, j] = constraints[row][col]
        
        b[constraint_idx] = rhs_values.get(row, 0.0)
        constraint_idx += 1
    
    # Set variable bounds
    lb = np.array([bounds[col]["LO"] for col in col_names])
    ub = np.array([bounds[col]["UP"] for col in col_names])
    
    # Split constraints by type
    split_start_time = time.time()
    print(f"Matrix setup completed in {split_start_time - matrix_start_time:.2f} seconds")
    print(f"Splitting constraints by type...")
    
    A_eq = []
    b_eq = []
    A_ineq = []
    b_ineq = []
    
    for i, row_type in enumerate(constraint_types):
        if row_type == 'E':  # Equality
            A_eq.append(A[i])
            b_eq.append(b[i])
        elif row_type == 'L':  # Less than
            A_ineq.append(A[i])
            b_ineq.append(b[i])
        elif row_type == 'G':  # Greater than
            A_ineq.append(-A[i])  # Negate to convert to ≤
            b_ineq.append(-b[i])
    
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None
    A_ineq = np.array(A_ineq) if A_ineq else None
    b_ineq = np.array(b_ineq) if b_ineq else None
    
    print(f"Constraint splitting completed in {time.time() - split_start_time:.2f} seconds")
    print(f"Total parsing time: {time.time() - parse_start_time:.2f} seconds")
    
    return {
        'c': c,
        'A_eq': A_eq,
        'b_eq': b_eq,
        'A_ineq': A_ineq,
        'b_ineq': b_ineq,
        'bounds': (lb, ub),
        'col_names': col_names,
        'n_vars': n_vars,
        'n_eq': len(b_eq) if b_eq is not None else 0,
        'n_ineq': len(b_ineq) if b_ineq is not None else 0,
    } 