import numpy as np
from scipy.sparse import csr_matrix
from typing import Optional, Tuple, List
from pydantic import BaseModel, Field, ConfigDict

# Define Pydantic model for LP data
class LpData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_vars: int
    c: np.ndarray
    bounds: Tuple[np.ndarray, np.ndarray]
    A_eq: Optional[csr_matrix] = None
    b_eq: np.ndarray = Field(default_factory=lambda: np.array([]))
    A_ineq: Optional[csr_matrix] = None
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
    
    # Similar validators can be added for bounds, b_eq, b_ineq, A_eq, A_ineq

    # Computed properties (optional, but can be useful)
    @property
    def n_eq(self) -> int:
        return self.A_eq.shape[0] if self.A_eq is not None else 0

    @property
    def n_ineq(self) -> int:
        return self.A_ineq.shape[0] if self.A_ineq is not None else 0 