# Imputer

`Imputer` is a Rust implementation of a data imputation utility inspired by Scikit-learn. It allows replacing missing values in a dataset using a specified imputation strategy. This implementation integrates with Python via PyO3 and Maturin.

---

## Class Definition

### `Imputer`
- **Fields**:
  - `strategy`: Imputation strategy, either `Mean` or `Constant(f64)`.
  - `impute_values`: Optional vector of computed imputation values for each column.

---

## Methods (callable from Python)

### `new(strategy: &str, value: Option<f64>) -> Self`
- **Description**: Creates a new instance of `Imputer` with the specified imputation strategy.
- **Parameters**:
  - `strategy`: The imputation strategy. Accepted values:
    - `"mean"`: Replace missing values with the mean of the column.
    - `"constant"`: Replace missing values with a constant value.
  - `value`: The constant value for the `"constant"` strategy. Ignored if the strategy is `"mean"`.
- **Returns**: `Imputer` object.

---

### `fit(data: PyReadonlyArray2<f64>) -> PyResult<()>`
- **Description**: Computes the imputation values for each column in the dataset.
- **Parameters**:
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`. Missing values should be represented as `NaN`.
- **Returns**: `Ok(())` on success, `Err(PyValueError)` if an error occurs.

---

### `transform(py: Python, data: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>>`
- **Description**: Imputes missing values in the input dataset using the precomputed values (assumes fit has already been called).
- **Parameters**:
  - `py`: Python GIL token.
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`.
- **Returns**: Imputed data as a 2D NumPy array.

---

### `fit_transform(py: Python, data: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>>`
- **Description**: Computes the imputation values and imputes missing values in the input dataset.
- **Parameters**:
  - `py`: Python GIL token.
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`.
- **Returns**: Imputed data as a 2D NumPy array.

---

## Internal Methods (Rust Implementation)

### `fit_helper(data: &DMatrix<Option<f64>>) -> Result<(), ImputerError>`
- **Description**: Computes the imputation values for each column based on the specified strategy.
- **Parameters**:
  - `data`: Dynamic matrix of optional values representing the dataset.
- **Returns**: `Ok(())` on success, `Err(ImputerError)` if a column contains only missing values.

---

### `transform_helper(data: &DMatrix<Option<f64>>) -> DMatrix<f64>`
- **Description**: Imputes missing values in the input data using precomputed imputation values.
- **Parameters**:
  - `data`: Dynamic matrix of optional values representing the dataset.
- **Returns**: Imputed data as a dynamic matrix.

---

### `fit_transform_helper(data: &DMatrix<Option<f64>>) -> Result<DMatrix<f64>, ImputerError>`
- **Description**: Combines `fit_helper` and `transform_helper` to compute and impute missing values.
- **Parameters**:
  - `data`: Dynamic matrix of optional values representing the dataset.
- **Returns**: Imputed data as a dynamic matrix, or an error if a column contains only missing values.

---

## Example Usage (Python)

```python
import numpy as np
from rustkit import Imputer

# Create and fit an imputer
imputer = Imputer("mean", None)
data = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
imputer.fit(data)

# Transform data
transformed = imputer.transform(data)

# Fit and transform in one step
fit_transformed = imputer.fit_transform(data)
```

## **Notes**

- The "mean" strategy computes the mean of non-missing values in each column.
- The "constant" strategy replaces missing values with a specified constant value.
- Missing values in the input data must be represented as `NaN` for compatibility with NumPy.
- Columns with all missing values will raise an `ImputerError` during fitting with the "mean" strategy.
