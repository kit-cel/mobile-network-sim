"""Type definitions for the RAN simulation."""

import numpy as np
from numpy.typing import NDArray

# Custom type aliases for NumPy arrays with specific dtypes
BoolArr = NDArray[np.bool_]
FloatArr = NDArray[np.floating]
Float32Arr = NDArray[np.float32]
Float64Arr = NDArray[np.float64]
IntArr = NDArray[np.int_]
