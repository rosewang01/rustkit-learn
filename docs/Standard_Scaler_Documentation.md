# StandardScaler

`StandardScaler` is a Rust implementation of a data preprocessing utility inspired by Scikit-learn. It standardizes features by removing the mean and scaling to unit variance. This implementation integrates with Python via PyO3 and Maturin. Each of the methods callable from Python makes use of a corresponding `_helper`  which actually implements that method in Rust.

---

## Class Definition

### `StandardScaler`
- **Fields**:
  - `means`: Optional vector of column-wise means.
  - `std_devs`: Optional vector of column-wise standard deviations.

---

## Methods (callable from Python)

### `new()`
- **Description**: Creates a new instance of `StandardScaler`.
- **Returns**: `StandardScaler` object.

---

### `fit(data: PyReadonlyArray2<f64>) -> PyResult<()>`
- **Description**: Computes the mean and standard deviation for each feature in the input dataset.
- **Parameters**:
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`.
- **Returns**: `Ok(())` on success, `Err(PyValueError)` if an error occurs.

---

### `transform(py: Python, data: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>>`
- **Description**: Standardizes the input data using the precomputed means and standard deviations.
- **Parameters**:
  - `py`: Python GIL token.
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`.
- **Returns**: Transformed data as a 2D NumPy array.

---

### `fit_transform(py: Python, data: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>>`
- **Description**: Computes the means and standard deviations, then standardizes the input data.
- **Parameters**:
  - `py`: Python GIL token.
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`.
- **Returns**: Transformed data as a 2D NumPy array.

---

### `inverse_transform(py: Python, scaled_data: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>>`
- **Description**: Reverts standardized data back to its original scale.
- **Parameters**:
  - `py`: Python GIL token.
  - `scaled_data`: 2D NumPy array of standardized data `(n_samples, n_features)`.
- **Returns**: Original data as a 2D NumPy array.

---

## Internal Methods (Rust implementation)

### `fit_helper(data: &DMatrix<f64>)`
- **Description**: Computes column-wise means and standard deviations from a dynamic matrix.
- **Parameters**:
  - `data`: Dynamic matrix representation of the dataset.

---

### `transform_helper(data: &DMatrix<f64>) -> DMatrix<f64>`
- **Description**: Standardizes the input data using precomputed means and standard deviations.
- **Parameters**:
  - `data`: Dynamic matrix representation of the dataset.
- **Returns**: Standardized data as a dynamic matrix.

---

### `fit_transform_helper(data: &DMatrix<f64>) -> DMatrix<f64>`
- **Description**: Combines `fit_helper` and `transform_helper` to compute and standardize data.
- **Parameters**:
  - `data`: Dynamic matrix representation of the dataset.
- **Returns**: Standardized data as a dynamic matrix.

---

### `inverse_transform_helper(scaled_data: &DMatrix<f64>) -> DMatrix<f64>`
- **Description**: Reverts standardized data back to its original scale.
- **Parameters**:
  - `scaled_data`: Dynamic matrix of standardized data.
- **Returns**: Original data as a dynamic matrix.

---

## Example Usage (Python)

```python
import numpy as np
from rustkit import StandardScaler

# Create and fit scaler
scaler = StandardScaler()
data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
scaler.fit(data)

# Transform data
transformed = scaler.transform(data)

# Fit and transform in one step
fit_transformed = scaler.fit_transform(data)

# Revert transformed data
original = scaler.inverse_transform(transformed)
```

---

## **Notes**

- This implementation currently computes standard deviation using n (population standard deviation). Future versions may add an option for n-1 (sample standard deviation).
