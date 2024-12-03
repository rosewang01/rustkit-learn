# R2Score

`R2Score` is a Rust implementation of the R² score (coefficient of determination) calculation. It evaluates the proportion of variance in the dependent variable that is predictable from the independent variable(s). This implementation integrates with Python via PyO3 and Maturin.

---

## Class Definition

### `R2Score`

`R2Score` provides methods to compute the R² score between true and predicted values.

---

## Methods (callable from Python)

### `compute(py: Python, y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>) -> PyResult<Py<PyFloat>>`
- **Description**: Computes the R² score between the true values (`y_true`) and the predicted values (`y_pred`).
- **Parameters**:
  - `py`: Python GIL token.
  - `y_true`: 1D NumPy array of true values.
  - `y_pred`: 1D NumPy array of predicted values.
- **Returns**: R² score as a Python float.
- **Panics**: Raises an error if `y_true` and `y_pred` have different lengths.

---

## Internal Methods (Rust implementation)

### `compute_helper(y_true: &DVector<f64>, y_pred: &DVector<f64>) -> f64`
- **Description**: Calculates the R² score given two dynamic vectors.
- **Parameters**:
  - `y_true`: Dynamic vector of true values.
  - `y_pred`: Dynamic vector of predicted values.
- **Returns**: R² score.
- **Panics**: Raises an error if `y_true` and `y_pred` have different lengths.

---

## Example Usage (Python)

```python
import numpy as np
from your_module import R2Score

# True and predicted values
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

# Compute R² score
r2_score = R2Score.compute(y_true, y_pred)
print(f"R² Score: {r2_score}")
