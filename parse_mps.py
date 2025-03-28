import numpy as np
import time
from scipy.sparse import coo_matrix, csr_matrix, vstack # Added vstack import here
from typing import Dict, Any, Tuple, List, Optional # Add typing
from lp_model import LpData # Import the model

def parse_mps(path: str) -> LpData: # Update return type hint
    """Parse an MPS file and return an LpData object."""
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
    n_constraints_total = len(row_names) - 1 # Excluding objective row
    
    # Create dictionaries for sparse matrix construction (COO format)
    row_ind = []
    col_ind = []
    data = []
    
    constraint_rhs = {}
    constraint_types = {}
    
    # Map names to indices
    col_to_idx = {name: i for i, name in enumerate(col_names)}
    row_to_idx = {}
    constraint_idx = 0
    for name in row_names:
        if name != objective_name:
            row_to_idx[name] = constraint_idx
            constraint_rhs[constraint_idx] = rhs_values.get(name, 0.0)
            constraint_types[constraint_idx] = row_types[name]
            constraint_idx += 1

    # Fill objective coefficients
    c = np.zeros(n_vars)
    for col_name, value in objective.items():
        if col_name in col_to_idx: # Check if column exists
            c[col_to_idx[col_name]] = value
    
    # Fill constraint matrix data in COO format
    for row_name, cols in constraints.items():
        if row_name == objective_name:
             continue # Should not happen based on previous logic, but safe check
        if row_name not in row_to_idx:
             print(f"Warning: Constraint row '{row_name}' found in COLUMNS but not in ROWS section. Skipping.")
             continue
        current_row_idx = row_to_idx[row_name]
        for col_name, value in cols.items():
            if col_name in col_to_idx:
                 current_col_idx = col_to_idx[col_name]
                 row_ind.append(current_row_idx)
                 col_ind.append(current_col_idx)
                 data.append(value)
            else:
                 print(f"Warning: Column '{col_name}' in constraint '{row_name}' not found in COLUMNS section list. Skipping coefficient.")

    # Set variable bounds
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, float('inf')) # Default upper bound is infinity
    for i, col in enumerate(col_names):
        if col in bounds:
            lb[i] = bounds[col]["LO"]
            ub[i] = bounds[col]["UP"]
        # Else defaults (lb=0, ub=inf) are already set

    print(f"Matrix data gathered in {time.time() - matrix_start_time:.2f} seconds")
    split_start_time = time.time()
    print(f"Building and splitting constraint matrix...")
    
    # Build the full sparse matrix (COO)
    if n_constraints_total > 0 and len(data) > 0:
        full_A_coo = coo_matrix((data, (row_ind, col_ind)), shape=(n_constraints_total, n_vars))
        full_A = full_A_coo.tocsr() # Convert to CSR for efficient row slicing
    else:
        # Create empty COO components if no constraints
        full_A = csr_matrix((n_constraints_total, n_vars))

    # Prepare indices for splitting
    eq_indices = [idx for idx, type in constraint_types.items() if type == 'E']
    l_indices = [idx for idx, type in constraint_types.items() if type == 'L']
    g_indices = [idx for idx, type in constraint_types.items() if type == 'G']

    n_eq = len(eq_indices)
    n_ineq = len(l_indices) + len(g_indices)
    
    # Initialize COO components for A_eq and A_ineq
    A_eq_row: Optional[np.ndarray] = None
    A_eq_col: Optional[np.ndarray] = None
    A_eq_data: Optional[np.ndarray] = None
    b_eq: np.ndarray = np.array([])
    
    A_ineq_row: Optional[np.ndarray] = None
    A_ineq_col: Optional[np.ndarray] = None
    A_ineq_data: Optional[np.ndarray] = None
    b_ineq: np.ndarray = np.array([])

    if n_eq > 0:
        A_eq_csr = full_A[eq_indices, :]
        A_eq_coo = A_eq_csr.tocoo()
        A_eq_row = A_eq_coo.row
        A_eq_col = A_eq_coo.col
        A_eq_data = A_eq_coo.data
        b_eq = np.array([constraint_rhs[idx] for idx in eq_indices])
        
    if n_ineq > 0:
        # Combine L and G constraints into a single inequality matrix (Ax <= b)
        rows_L_csr = full_A[l_indices, :] if l_indices else None
        rhs_L = np.array([constraint_rhs[idx] for idx in l_indices]) if l_indices else np.array([])
        
        rows_G_csr = -full_A[g_indices, :] if g_indices else None # Negate G constraints
        rhs_G = np.array([-constraint_rhs[idx] for idx in g_indices]) if g_indices else np.array([]) # Negate G rhs
        
        # Stack L and G vertically if both exist
        A_ineq_csr: Optional[csr_matrix] = None
        if rows_L_csr is not None and rows_G_csr is not None:
            A_ineq_csr = vstack([rows_L_csr, rows_G_csr], format='csr')
            b_ineq = np.concatenate([rhs_L, rhs_G])
        elif rows_L_csr is not None:
            A_ineq_csr = rows_L_csr
            b_ineq = rhs_L
        elif rows_G_csr is not None:
            A_ineq_csr = rows_G_csr
            b_ineq = rhs_G

        if A_ineq_csr is not None:
            A_ineq_coo = A_ineq_csr.tocoo()
            A_ineq_row = A_ineq_coo.row
            A_ineq_col = A_ineq_coo.col
            A_ineq_data = A_ineq_coo.data
        # Else: A_ineq components remain None, b_ineq remains empty if n_ineq was > 0 but no rows resulted
        
    print(f"Constraint splitting completed in {time.time() - split_start_time:.2f} seconds")
    print(f"Total parsing time: {time.time() - parse_start_time:.2f} seconds")
    
    # Create and return LpData instance with COO components
    try:
        lp_data_obj = LpData(
            n_vars=n_vars,
            c=c,
            bounds=(lb, ub),
            A_eq_row=A_eq_row, 
            A_eq_col=A_eq_col,
            A_eq_data=A_eq_data,
            b_eq=b_eq,
            A_ineq_row=A_ineq_row,
            A_ineq_col=A_ineq_col,
            A_ineq_data=A_ineq_data,
            b_ineq=b_ineq,
            obj_offset=0.0, # Default obj_offset
            col_names=col_names,
        )
        return lp_data_obj
    except Exception as e:
        print(f"Error creating LpData object in parse_mps: {e}")
        # Handle error appropriately, maybe raise it or return a default/error state
        raise e 