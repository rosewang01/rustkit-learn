# PCA

`PCA` (Principal Component Analysis) is a Rust implementation of a dimensionality reduction algorithm. It reduces the dimensionality of the data while preserving as much variability as possible. This implementation integrates with Python via PyO3 and Maturin.

---

## Class Definition

### `PCA`
- **Fields**:
  - `components`: Matrix representing the principal components.
  - `explained_variance`: Vector of explained variances corresponding to each principal component.
  - `mean`: Row vector of feature means.

---

## Methods (callable from Python)

### `new()`
- **Description**: Creates a new instance of `PCA`.
- **Returns**: `PCA` object.

---

### `fit(data: PyReadonlyArray2<f64>, n_components: i64) -> PyResult<()>`
- **Description**: Fits the PCA model to the data, computing principal components and explained variance.
- **Parameters**:
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`.
  - `n_components`: Number of principal components to compute.
- **Returns**: `Ok(())` on success, `Err(PyValueError)` if an error occurs.

---

### `transform(py: Python, data: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>>`
- **Description**: Transforms the data into the principal component space.
- **Parameters**:
  - `py`: Python GIL token.
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`.
- **Returns**: Transformed data as a 2D NumPy array.

---

### `fit_transform(py: Python, data: PyReadonlyArray2<f64>, n_components: i64) -> PyResult<Py<PyArray2<f64>>>`
- **Description**: Combines fitting the PCA model and transforming the data in one step.
- **Parameters**:
  - `py`: Python GIL token.
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`.
  - `n_components`: Number of principal components to compute.
- **Returns**: Transformed data as a 2D NumPy array.

---

### `inverse_transform(py: Python, data: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>>`
- **Description**: Reconstructs the original data from its projection in the principal component space.
- **Parameters**:
  - `py`: Python GIL token.
  - `data`: 2D NumPy array of shape `(n_samples, n_components)`.
- **Returns**: Reconstructed data as a 2D NumPy array.

---

### Getters

#### `components(py: Python) -> PyResult<Py<PyArray2<f64>>>`
- **Description**: Returns the principal components.
- **Returns**: 2D NumPy array of shape `(n_features, n_components)`.

#### `explained_variance(py: Python) -> PyResult<Py<PyArray1<f64>>>`
- **Description**: Returns the explained variance for each principal component.
- **Returns**: 1D NumPy array.

#### `mean(py: Python) -> PyResult<Py<PyArray1<f64>>>`
- **Description**: Returns the mean of each feature.
- **Returns**: 1D NumPy array.

---

### Setters

#### `set_components(components: PyReadonlyArray2<f64>) -> PyResult<()>`
- **Description**: Sets the principal components.
- **Parameters**:
  - `components`: 2D NumPy array of principal components.
- **Returns**: `Ok(())`.

#### `set_explained_variance(explained_variance: PyReadonlyArray1<f64>) -> PyResult<()>`
- **Description**: Sets the explained variance for each principal component.
- **Parameters**:
  - `explained_variance`: 1D NumPy array.
- **Returns**: `Ok(())`.

#### `set_mean(mean: PyReadonlyArray1<f64>) -> PyResult<()>`
- **Description**: Sets the mean of each feature.
- **Parameters**:
  - `mean`: 1D NumPy array.
- **Returns**: `Ok(())`.

---

## Internal Methods (Rust Implementation)

### `fit_helper(data: &DMatrix<f64>, n_components: usize)`
- **Description**: Computes the principal components and explained variance.
- **Parameters**:
  - `data`: Dynamic matrix of shape `(n_samples, n_features)`.
  - `n_components`: Number of principal components to compute.

---

### `transform_helper(data: &DMatrix<f64>) -> DMatrix<f64>`
- **Description**: Transforms the data into the principal component space.
- **Parameters**:
  - `data`: Dynamic matrix of shape `(n_samples, n_features)`.
- **Returns**: Transformed data as a dynamic matrix.

---

### `fit_transform_helper(data: &DMatrix<f64>, n_components: usize) -> DMatrix<f64>`
- **Description**: Combines fitting the PCA model and transforming the data.
- **Parameters**:
  - `data`: Dynamic matrix of shape `(n_samples, n_features)`.
  - `n_components`: Number of principal components to compute.
- **Returns**: Transformed data as a dynamic matrix.

---

### `inverse_transform_helper(data: &DMatrix<f64>) -> DMatrix<f64>`
- **Description**: Reconstructs the original data from its projection in the principal component space.
- **Parameters**:
  - `data`: Dynamic matrix of shape `(n_samples, n_components)`.
- **Returns**: Reconstructed data as a dynamic matrix.

---

## Example Usage (Python)

```python
import numpy as np
from ruskit import PCA

# Create and fit PCA
pca = PCA()
data = np.random.rand(100, 5)
pca.fit(data, n_components=2)

# Transform data
transformed = pca.transform(data)

# Fit and transform in one step
fit_transformed = pca.fit_transform(data, n_components=2)

# Reconstruct data
reconstructed = pca.inverse_transform(transformed)
```

---

## **Notes**

- The implementation currently uses full Singular Value Decomposition (SVD), which may be inefficient for large datasets. Future versions may incorporate partial SVD.