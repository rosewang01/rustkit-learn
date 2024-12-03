# KMeans

`KMeans` is a Rust implementation of the K-Means clustering algorithm, including support for K-Means++ and random initialization methods. This implementation integrates with Python via PyO3 and Maturin.

---

## Class Definition

### `KMeans`
- **Fields**:
  - `k`: Number of clusters.
  - `max_iter`: Maximum number of iterations per run.
  - `n_init`: Number of runs to select the best clustering result.
  - `init_method`: Initialization method (`KMeansPlusPlus` or `Random`).
  - `centroids`: Centroids of the clusters after fitting (optional).
  - `labels`: Cluster labels for each data point after fitting (optional).

---

## Methods (callable from Python)

### `new(k: usize, init_method_str: &str, max_iter: Option<usize>, n_init: Option<usize>) -> Self`
- **Description**: Creates a new `KMeans` instance with the specified parameters.
- **Parameters**:
  - `k`: Number of clusters.
  - `init_method_str`: Initialization method (`"kmeans++"` or `"random"`).
  - `max_iter`: Maximum number of iterations (default: 200).
  - `n_init`: Number of runs to initialize centroids - default depends on `init_method` (10 for random initialization, 1 for KMeans++)
- **Returns**: A new `KMeans` instance.

---

### `fit(data: PyReadonlyArray2<f64>) -> PyResult<()>`
- **Description**: Fits the KMeans model to the input data by computing centroids and assigning cluster labels.
- **Parameters**:
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`.
- **Returns**: `Ok(())` on success, `Err(PyValueError)` if an error occurs.

---

### `fit_predict(py: Python, data: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<usize>>>`
- **Description**: Fits the model and returns cluster labels for the input data.
- **Parameters**:
  - `py`: Python GIL token.
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`.
- **Returns**: Cluster labels as a 1D NumPy array.

---

### `compute_inertia(py: Python, data: PyReadonlyArray2<f64>, labels: PyReadonlyArray1<usize>) -> PyResult<Py<PyFloat>>`
- **Description**: Computes the inertia (sum of squared distances to the nearest centroids).
- **Parameters**:
  - `py`: Python GIL token.
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`.
  - `labels`: Cluster labels as a 1D NumPy array.
- **Returns**: Inertia as a floating-point value.

---

### `predict(py: Python, data: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<usize>>>`
- **Description**: Predicts cluster labels for new data points using the fitted centroids.
- **Parameters**:
  - `py`: Python GIL token.
  - `data`: 2D NumPy array of shape `(n_samples, n_features)`.
- **Returns**: Predicted labels as a 1D NumPy array.

---

### `centroids(py: Python) -> PyResult<Py<PyArray2<f64>>>`
- **Description**: Retrieves the computed centroids of the clusters.
- **Returns**: Centroids as a 2D NumPy array of shape `(n_features, k)`.

---

## Internal Methods (Rust Implementation)

### `fit_helper(data: &DMatrix<f64>)`
- **Description**: Fits the KMeans model to the input data for `n_init` iterations and keeps the best result.
- **Parameters**:
  - `data`: Dynamic matrix representation of the dataset.

---

### `fit_predict_helper(data: &DMatrix<f64>) -> DVector<usize>`
- **Description**: Combines `fit_helper` and `predict_helper` to fit the model and predict cluster labels in one step.
- **Parameters**:
  - `data`: Dynamic matrix representation of the dataset.
- **Returns**: Cluster labels as a dynamic vector.

---

### `compute_inertia_helper(data: &DMatrix<f64>, labels: &DVector<usize>, centroids: &DMatrix<f64>) -> f64`
- **Description**: Computes the inertia for the given data, labels, and centroids.
- **Parameters**:
  - `data`: Dynamic matrix representation of the dataset.
  - `labels`: Dynamic vector of cluster labels.
  - `centroids`: Dynamic matrix of centroids.
- **Returns**: Inertia as a floating-point value.

---

### `predict_helper(data: &DMatrix<f64>) -> Option<DVector<usize>>`
- **Description**: Predicts cluster labels for input data using the fitted centroids.
- **Parameters**:
  - `data`: Dynamic matrix representation of the dataset.
- **Returns**: Cluster labels as an optional dynamic vector.

---

### `get_centroids_helper() -> Option<&DMatrix<f64>>`
- **Description**: Retrieves the centroids of the clusters.
- **Returns**: Centroids as an optional dynamic matrix.

---

### `run_single(data: &DMatrix<f64>) -> (DMatrix<f64>, DVector<usize>, f64)`
- **Description**: Runs a single iteration of the KMeans algorithm.
- **Returns**: A tuple containing centroids, labels, and inertia.

---

### `random_init(data: &DMatrix<f64>) -> DMatrix<f64>`
- **Description**: Initializes centroids by randomly selecting data points.
- **Parameters**:
  - `data`: Dynamic matrix representation of the dataset.
- **Returns**: Centroids as a dynamic matrix.

---

### `kmeans_plus_plus(data: &DMatrix<f64>) -> DMatrix<f64>`
- **Description**: Initializes centroids using the KMeans++ method.
- **Parameters**:
  - `data`: Dynamic matrix representation of the dataset.
- **Returns**: Centroids as a dynamic matrix.

---

### `assign_labels(data: &DMatrix<f64>, centroids: &DMatrix<f64>) -> DVector<usize>`
- **Description**: Assigns each data point to the nearest centroid.
- **Returns**: Cluster labels as a dynamic vector.

---

### `update_centroids(data: &DMatrix<f64>, labels: &DVector<usize>) -> DMatrix<f64>`
- **Description**: Updates centroids based on the mean of assigned points.
- **Returns**: Updated centroids as a dynamic matrix.

---

## Example Usage (Python)

```python
import numpy as np
from ruskit import KMeans

# Create KMeans instance
kmeans = KMeans(k=3, init_method_str="kmeans++", max_iter=100, n_init=5)

# Fit model
data = np.random.rand(100, 2)
kmeans.fit(data)

# Predict cluster labels
labels = kmeans.predict(data)

# Fit and predict in one step
labels = kmeans.fit_predict(data)

# Compute inertia
inertia = kmeans.compute_inertia(data, labels)

# Access centroids
centroids = kmeans.centroids()
```

---

## **Notes**

- Centroids are stored as columns in the centroids matrix.