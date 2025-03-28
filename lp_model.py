import numpy as np
# Remove csr_matrix import if no longer needed elsewhere
# from scipy.sparse import csr_matrix 
from typing import Optional, Tuple, List
from pydantic import BaseModel, Field, ConfigDict

# Define Pydantic model for LP data
class LpData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_vars: int
    c: np.ndarray
    bounds: Tuple[np.ndarray, np.ndarray]
    # Replace A_eq with COO components
    A_eq_row: Optional[np.ndarray] = Field(default=None)
    A_eq_col: Optional[np.ndarray] = Field(default=None)
    A_eq_data: Optional[np.ndarray] = Field(default=None)
    b_eq: np.ndarray = Field(default_factory=lambda: np.array([]))
    # Replace A_ineq with COO components
    A_ineq_row: Optional[np.ndarray] = Field(default=None)
    A_ineq_col: Optional[np.ndarray] = Field(default=None)
    A_ineq_data: Optional[np.ndarray] = Field(default=None)
    b_ineq: np.ndarray = Field(default_factory=lambda: np.array([]))
    obj_offset: float = 0.0
    col_names: Optional[List[str]] = None

    # Validate that numpy arrays are correctly shaped/typed if needed
    # (Example validator - can be expanded)
    # @validator('c')
    # def check_c_shape(cls, v, values):
    #     if 'n_vars' in values and v.shape != (values['n_vars'],):
    #         raise ValueError(f"Shape mismatch for c: expected ({values['n_vars']},), got {v.shape}")
    #     if v.dtype != np.float64: # Example type check
    #          print(f"Warning: c dtype is {v.dtype}, converting to float64")
    #          return v.astype(np.float64)
    #     return v
    
    # Similar validators can be added for bounds, b_eq, b_ineq, etc.

    # Computed properties based on the length of the RHS vectors
    @property
    def n_eq(self) -> int:
        # n_eq is the number of equality constraints, typically the length of b_eq
        return self.b_eq.shape[0]

    @property
    def n_ineq(self) -> int:
         # n_ineq is the number of inequality constraints, typically the length of b_ineq
        return self.b_ineq.shape[0] 