# MSE

`MSE` is a Rust implementation of the mean squared error (MSE) calculation. It quantifies the average squared difference between true and predicted values, serving as a measure of model accuracy. This implementation integrates with Python via PyO3 and Maturin.

---

## Class Definition

### `MSE`

`MSE` provides methods to compute the mean squared error between true and predicted values.

---

## Methods (callable from Python)

### `compute(py: Python, y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>) -> PyResult<Py<PyFloat>>`
- **Description**: Computes the MSE between the true values (`y_true`) and the predicted values (`y_pred`).
- **Parameters**:
  - `py`: Python GIL token.
  - `y_true`: 1D NumPy array of true values.
  - `y_pred`: 1D NumPy array of predicted values.
- **Returns**: MSE as a Python float.
- **Panics**: Raises an error if `y_true` and `y_pred` have different lengths.

---

## Internal Methods (Rust implementation)

### `compute_helper(y_true: &DVector<f64>, y_pred: &DVector<f64>) -> f64`
- **Description**: Calculates the MSE given two dynamic vectors.
- **Parameters**:
  - `y_true`: Dynamic vector of true values.
  - `y_pred`: Dynamic vector of predicted values.
- **Returns**: MSE.
- **Panics**: Raises an error if `y_true` and `y_pred` have different lengths.

---

## Example Usage (Python)

```python
import numpy as np
from ruskit import MSE

# True and predicted values
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

# Compute MSE
mse = MSE.compute(y_true, y_pred)
print(f"MSE: {mse}")
