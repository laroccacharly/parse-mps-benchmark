import numpy as np
import time
from scipy.sparse import coo_matrix, csr_matrix, vstack
from typing import Optional, Dict, List, Tuple, Any
from lp_model import LpData
from dataclasses import dataclass, field

TIMEOUT_SECONDS = 1000

@dataclass
class _ParserState:
    """Internal state for the MPS parser."""
    row_names: List[str] = field(default_factory=list)
    col_names: List[str] = field(default_factory=list)
    objective_name: Optional[str] = None
    constraints: Dict[str, Dict[str, float]] = field(default_factory=dict)
    objective: Dict[str, float] = field(default_factory=dict)
    rhs_values: Dict[str, float] = field(default_factory=dict)
    bounds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    row_types: Dict[str, str] = field(default_factory=dict) # N, E, L, G

def _initialize_parser_state() -> _ParserState:
    """Initializes the parser state."""
    return _ParserState()

def _parse_rows_section(line: str, state: _ParserState):
    """Parses a line from the ROWS section."""
    parts = line.split()
    row_type = parts[0]
    row_name = parts[1]
    state.row_names.append(row_name)
    state.row_types[row_name] = row_type
    if row_type == 'N':
        state.objective_name = row_name

def _parse_columns_section(line: str, state: _ParserState):
    """Parses a line from the COLUMNS section."""
    # Skip markers
    if "'MARKER'" in line:
        return

    parts = line.split()
    col_name = parts[0]

    if col_name not in state.col_names:
        state.col_names.append(col_name)

    # Handle entries with multiple coefficients on a line
    i = 1
    while i < len(parts):
        row_name = parts[i]
        value = float(parts[i+1])

        if row_name == state.objective_name:
            state.objective[col_name] = value
        else:
            if row_name not in state.constraints:
                state.constraints[row_name] = {}
            state.constraints[row_name][col_name] = value
        i += 2

def _parse_rhs_section(line: str, state: _ParserState):
    """Parses a line from the RHS section."""
    parts = line.split()
    # Skip RHS name (parts[0])

    # Process each rhs entry
    i = 1
    while i < len(parts):
        row_name = parts[i]
        value = float(parts[i+1])
        state.rhs_values[row_name] = value
        i += 2

def _parse_bounds_section(line: str, state: _ParserState):
    """Parses a line from the BOUNDS section."""
    parts = line.split()
    bound_type = parts[0]
    # Skip bound name (parts[1])
    col_name = parts[2]
    value = float(parts[3]) if len(parts) > 3 else 0.0

    if col_name not in state.bounds:
        # Default bounds [0, +inf] before applying specific bound
        state.bounds[col_name] = {"LO": 0.0, "UP": float('inf')}

    if bound_type == 'LO':
        state.bounds[col_name]["LO"] = value
    elif bound_type == 'UP':
        state.bounds[col_name]["UP"] = value
    elif bound_type == 'FX':
        state.bounds[col_name]["LO"] = value
        state.bounds[col_name]["UP"] = value
    elif bound_type == 'FR':
        state.bounds[col_name]["LO"] = float('-inf')
        state.bounds[col_name]["UP"] = float('inf')
    elif bound_type == 'MI':
        state.bounds[col_name]["LO"] = float('-inf')
        # UP remains default or previous value
    elif bound_type == 'PL':
        # LO remains default or previous value
        state.bounds[col_name]["UP"] = float('inf')
    elif bound_type == 'BV': # Binary variable
        state.bounds[col_name]["LO"] = 0.0
        state.bounds[col_name]["UP"] = 1.0
    # Note: UI (Upper Integer) and LI (Lower Integer) are ignored for continuous LP


def _set_default_bounds(state: _ParserState):
    """Sets default bounds [0, inf] for variables without explicit bounds."""
    for col in state.col_names:
        if col not in state.bounds:
            state.bounds[col] = {"LO": 0.0, "UP": float('inf')}


def _create_variable_bounds(state: _ParserState) -> Tuple[np.ndarray, np.ndarray]:
    """Creates lower and upper bound vectors from the parser state."""
    n_vars = len(state.col_names)
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, float('inf')) # Default upper bound is infinity
    for i, col in enumerate(state.col_names):
        if col in state.bounds:
            lb[i] = state.bounds[col].get("LO", 0.0) # Use .get for safety
            ub[i] = state.bounds[col].get("UP", float('inf'))
    return lb, ub

def _build_matrices_and_vectors(state: _ParserState) -> Dict[str, Any]:
    """Builds objective vector, constraint matrices (COO), and RHS vectors."""
    matrix_start_time = time.time()
    print(f"Starting matrix conversion...")

    n_vars = len(state.col_names)
    n_constraints_total = len(state.row_names) - (1 if state.objective_name else 0)

    # Map names to indices
    col_to_idx = {name: i for i, name in enumerate(state.col_names)}
    row_to_idx = {}
    constraint_idx = 0
    constraint_rhs_map = {} # Map original row name to rhs value
    constraint_type_map = {} # Map original row name to type
    constraint_rhs_indexed = {} # Map constraint index to rhs value
    constraint_types_indexed = {} # Map constraint index to type


    for name in state.row_names:
        if name != state.objective_name:
            row_to_idx[name] = constraint_idx
            constraint_rhs_map[name] = state.rhs_values.get(name, 0.0)
            constraint_type_map[name] = state.row_types.get(name, 'L') # Default? Or error?
            constraint_rhs_indexed[constraint_idx] = constraint_rhs_map[name]
            constraint_types_indexed[constraint_idx] = constraint_type_map[name]
            constraint_idx += 1

    # Fill objective coefficients
    c = np.zeros(n_vars)
    for col_name, value in state.objective.items():
        if col_name in col_to_idx: # Check if column exists
            c[col_to_idx[col_name]] = value

    # Fill constraint matrix data in COO format
    row_ind: List[int] = []
    col_ind: List[int] = []
    data: List[float] = []
    for row_name, cols in state.constraints.items():
        if row_name == state.objective_name:
             continue
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

    print(f"Matrix data gathered in {time.time() - matrix_start_time:.2f} seconds")
    split_start_time = time.time()
    print(f"Building and splitting constraint matrix...")

    # Build the full sparse matrix (COO)
    if n_constraints_total > 0 and len(data) > 0:
        full_A_coo = coo_matrix((data, (row_ind, col_ind)), shape=(n_constraints_total, n_vars))
        full_A = full_A_coo.tocsr() # Convert to CSR for efficient row slicing
    else:
        # Create empty CSR matrix if no constraints or no data
        full_A = csr_matrix((n_constraints_total, n_vars))

    # Prepare indices for splitting based on indexed types
    eq_indices = [idx for idx, type_val in constraint_types_indexed.items() if type_val == 'E']
    l_indices = [idx for idx, type_val in constraint_types_indexed.items() if type_val == 'L']
    g_indices = [idx for idx, type_val in constraint_types_indexed.items() if type_val == 'G']

    n_eq = len(eq_indices)
    n_ineq = len(l_indices) + len(g_indices)

    # Initialize COO components and RHS vectors
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
        b_eq = np.array([constraint_rhs_indexed[idx] for idx in eq_indices])

    if n_ineq > 0:
        rows_L_csr = full_A[l_indices, :] if l_indices else None
        rhs_L = np.array([constraint_rhs_indexed[idx] for idx in l_indices]) if l_indices else np.array([])

        rows_G_csr = -full_A[g_indices, :] if g_indices else None # Negate G constraints
        rhs_G = np.array([-constraint_rhs_indexed[idx] for idx in g_indices]) if g_indices else np.array([]) # Negate G rhs

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

    print(f"Constraint splitting completed in {time.time() - split_start_time:.2f} seconds")

    return {
        "n_vars": n_vars,
        "c": c,
        "A_eq_row": A_eq_row,
        "A_eq_col": A_eq_col,
        "A_eq_data": A_eq_data,
        "b_eq": b_eq,
        "A_ineq_row": A_ineq_row,
        "A_ineq_col": A_ineq_col,
        "A_ineq_data": A_ineq_data,
        "b_ineq": b_ineq,
    }


def parse_mps(path: str) -> LpData:
    """Parse an MPS file and return an LpData object."""
    parse_start_time = time.time()
    print(f"Starting MPS parsing for file: {path}")

    state = _initialize_parser_state()
    current_section = None

    section_parsers = {
        'ROWS': _parse_rows_section,
        'COLUMNS': _parse_columns_section,
        'RHS': _parse_rhs_section,
        'BOUNDS': _parse_bounds_section,
    }

    try:
        with open(path, 'r') as f:
            for line_num, line in enumerate(f):
                # Check for timeout periodically
                if line_num % 100 == 0:
                     current_time = time.time()
                     if current_time - parse_start_time > TIMEOUT_SECONDS:
                         raise TimeoutError(f"MPS parsing exceeded timeout of {TIMEOUT_SECONDS} seconds.")

                line = line.strip()
                if not line or line.startswith('*'):
                    continue

                # Identify section headers
                if line in ['NAME', 'ROWS', 'COLUMNS', 'RHS', 'RANGES', 'BOUNDS', 'ENDATA']:
                    if line == 'ENDATA':
                        break # Stop parsing at ENDATA
                    current_section = line
                    if line == 'RANGES':
                        print("Warning: RANGES section is not currently handled by this parser.")
                    continue

                # Parse line based on current section
                if current_section in section_parsers:
                    try:
                        section_parsers[current_section](line, state)
                    except Exception as e:
                        print(f"Error parsing line {line_num+1} in section {current_section}: {line}")
                        raise ValueError(f"Error parsing line {line_num+1} in section {current_section}: {e}") from e
                elif current_section == 'NAME':
                    # Typically the model name is on the same line, ignore for now
                    pass # Or store the name if needed later

    except FileNotFoundError:
        print(f"Error: MPS file not found at {path}")
        raise
    except TimeoutError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during file reading or section parsing: {e}")
        raise


    print(f"Finished reading MPS sections in {time.time() - parse_start_time:.2f} seconds")

    # Post-processing and matrix construction
    _set_default_bounds(state)
    lb, ub = _create_variable_bounds(state)
    matrix_data = _build_matrices_and_vectors(state)

    print(f"Total parsing time: {time.time() - parse_start_time:.2f} seconds")

    # Create and return LpData instance
    try:
        lp_data_obj = LpData(
            n_vars=matrix_data["n_vars"],
            c=matrix_data["c"],
            bounds=(lb, ub),
            A_eq_row=matrix_data["A_eq_row"],
            A_eq_col=matrix_data["A_eq_col"],
            A_eq_data=matrix_data["A_eq_data"],
            b_eq=matrix_data["b_eq"],
            A_ineq_row=matrix_data["A_ineq_row"],
            A_ineq_col=matrix_data["A_ineq_col"],
            A_ineq_data=matrix_data["A_ineq_data"],
            b_ineq=matrix_data["b_ineq"],
            obj_offset=0.0, # Default obj_offset, MPS doesn't typically specify this directly
            col_names=state.col_names, # Pass column names
        )
        print(f"Successfully created LpData object.")
        return lp_data_obj
    except Exception as e:
        print(f"Error creating LpData object in parse_mps: {e}")
        # Potentially log more details about the state or matrix_data here
        raise RuntimeError(f"Failed to construct LpData object: {e}") from e 