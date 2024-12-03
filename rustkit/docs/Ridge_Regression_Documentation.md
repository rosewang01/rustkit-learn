# RidgeRegression

`RidgeRegression` is a Rust implementation of the Ridge Regression algorithm. This model minimizes the residual sum of squares with L2 regularization, allowing for improved generalization by penalizing large coefficients. The implementation supports Python interop using PyO3 and Maturin.

---

## Class Definition

### `RidgeRegression`
- **Fields**:
  - `weights`: Vector of model weights (coefficients).
  - `intercept`: Optional bias term (intercept) for the model.
  - `regularization`: Regularization strength (λ). Set to `0.0` for standard linear regression.
  - `with_bias`: Boolean flag indicating whether a bias term is included.

---

## Methods (callable from Python)

### `new(regularization: f64, with_bias: bool) -> Self`
- **Description**: Creates a new instance of `RidgeRegression`.
- **Parameters**:
  - `regularization`: L2 regularization strength. Set to `0.0` for standard linear regression.
  - `with_bias`: Whether to include a bias term in the model.
- **Returns**: `RidgeRegression` object.

---

### `weights(py: Python) -> PyResult<Py<PyArray1<f64>>>`
- **Description**: Retrieves the model weights (coefficients).
- **Parameters**:
  - `py`: Python GIL token.
- **Returns**: Weights as a 1D NumPy array.

---

### `intercept(py: Python) -> PyResult<Py<PyFloat>>`
- **Description**: Retrieves the model intercept (bias term), if applicable.
- **Parameters**:
  - `py`: Python GIL token.
- **Returns**: Intercept as a Python float or `None`.

---

### `set_regularization(regularization: f64) -> PyResult<()>`
- **Description**: Updates the regularization strength.
- **Parameters**:
  - `regularization`: New L2 regularization strength.
- **Returns**: `Ok(())` on success.

---

### `set_with_bias(with_bias: bool) -> PyResult<()>`
- **Description**: Updates the inclusion of a bias term.
- **Parameters**:
  - `with_bias`: Whether to include a bias term in the model.
- **Returns**: `Ok(())` on success.

---

### `set_weights(weights: PyReadonlyArray1<f64>) -> PyResult<()>`
- **Description**: Sets custom model weights.
- **Parameters**:
  - `weights`: 1D NumPy array of weights.
- **Returns**: `Ok(())` on success.

---

### `set_intercept(intercept: Option<f64>) -> PyResult<()>`
- **Description**: Sets a custom model intercept.
- **Parameters**:
  - `intercept`: Optional intercept value.
- **Returns**: `Ok(())` on success.

---

### `fit(data: PyReadonlyArray2<f64>, target: PyReadonlyArray1<f64>) -> PyResult<()>`
- **Description**: Fits the Ridge Regression model to the provided data.
- **Parameters**:
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`.
  - `target`: 1D NumPy array of target values of shape `(n_samples,)`.
- **Returns**: `Ok(())` on success, `Err(PyValueError)` if an error occurs.

---

### `predict(py: Python, data: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>>`
- **Description**: Predicts target values for the provided input data.
- **Parameters**:
  - `py`: Python GIL token.
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`.
- **Returns**: Predictions as a 1D NumPy array.

---

## Internal Methods (Rust Implementation)

### `fit_helper(x: &DMatrix<f64>, y: &DVector<f64>)`
- **Description**: Performs Ridge Regression fitting using LU decomposition to solve for weights and intercept.
- **Parameters**:
  - `x`: Dynamic matrix of input data.
  - `y`: Dynamic vector of target values.

---

### `predict_helper(x: &DMatrix<f64>) -> DVector<f64>`
- **Description**: Computes predictions using the fitted model weights and intercept.
- **Parameters**:
  - `x`: Dynamic matrix of input data.
- **Returns**: Dynamic vector of predictions.

---

### `weights_helper() -> &DVector<f64>`
- **Description**: Retrieves the model weights (coefficients).
- **Returns**: Reference to the weights vector.

---

### `intercept_helper() -> Option<f64>`
- **Description**: Retrieves the model intercept (bias term), if available.
- **Returns**: Optional intercept value.

---

## Example Usage (Python)

```python
import numpy as np
from your_module import RidgeRegression

# Create Ridge Regression model
ridge = RidgeRegression(regularization=1.0, with_bias=True)

# Example data
X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
y = np.array([1.0, 2.0, 3.0])

# Fit the model
ridge.fit(X, y)

# Make predictions
predictions = ridge.predict(X)

# Access weights and intercept
weights = ridge.weights
intercept = ridge.intercept

```

---

## **Notes**

- This implementation assumes that input data is normalized. If not normalized, set regularization = 0.0 for correct results.
- Uses LU decomposition for efficient computation instead of directly inverting the matrix.