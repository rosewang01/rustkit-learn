use crate::benchmarking::log_function_time;
use crate::converters::{
    python_to_rust_dynamic_matrix, python_to_rust_dynamic_vector, rust_to_python_dynamic_matrix,
    rust_to_python_dynamic_vector,
};
use nalgebra::{linalg::SVD, DMatrix, DVector, RowDVector};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
pub struct PCA {
    components: DMatrix<f64>,
    explained_variance: DVector<f64>,
    mean: RowDVector<f64>,
}

// TODO: make a fix to use partial SVD decomposition for efficiency
#[pymethods]
impl PCA {
    /// Creates a new PCA instance.
    #[new]
    pub fn new() -> Self {
        PCA {
            components: DMatrix::zeros(0, 0),
            explained_variance: DVector::zeros(0),
            mean: RowDVector::zeros(0),
        }
    }

    pub fn fit(&mut self, data: PyReadonlyArray2<f64>, n_components: i64) -> PyResult<()> {
        let x = python_to_rust_dynamic_matrix(&data);
        let result = log_function_time(
            || self.fit_helper(&x, n_components.abs() as usize),
            "PCA::fit",
            x.nrows(),
            x.ncols(),
        );
        match result {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }

    pub fn transform(
        &self,
        py: Python,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let x = python_to_rust_dynamic_matrix(&data);
        let transformed_data = log_function_time(
            || self.transform_helper(&x),
            "PCA::transform",
            x.nrows(),
            x.ncols(),
        )
        .unwrap();
        rust_to_python_dynamic_matrix(py, transformed_data)
    }

    pub fn fit_transform(
        &mut self,
        py: Python,
        data: PyReadonlyArray2<f64>,
        n_components: i64,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let x = python_to_rust_dynamic_matrix(&data);
        let transformed_data = log_function_time(
            || self.fit_transform_helper(&x, n_components.abs() as usize),
            "PCA::fit_transform",
            x.nrows(),
            x.ncols(),
        )
        .unwrap();
        rust_to_python_dynamic_matrix(py, transformed_data)
    }

    pub fn inverse_transform(
        &self,
        py: Python,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let x = python_to_rust_dynamic_matrix(&data);
        let original_data = log_function_time(
            || self.inverse_transform_helper(&x),
            "PCA::inverse_transform",
            x.nrows(),
            x.ncols(),
        )
        .unwrap();
        rust_to_python_dynamic_matrix(py, original_data)
    }

    #[getter]
    pub fn components(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        rust_to_python_dynamic_matrix(py, self.components.clone())
    }

    #[getter]
    pub fn explained_variance(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        rust_to_python_dynamic_vector(py, self.explained_variance.clone())
    }

    #[getter]
    pub fn mean(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        let mean = self.mean.transpose();
        rust_to_python_dynamic_vector(py, mean)
    }

    #[setter]
    pub fn set_components(&mut self, components: PyReadonlyArray2<f64>) -> PyResult<()> {
        self.components = python_to_rust_dynamic_matrix(&components);
        Ok(())
    }

    #[setter]
    pub fn set_explained_variance(
        &mut self,
        explained_variance: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        self.explained_variance = python_to_rust_dynamic_vector(&explained_variance);
        Ok(())
    }

    #[setter]
    pub fn set_mean(&mut self, mean: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.mean = python_to_rust_dynamic_vector(&mean).transpose();
        Ok(())
    }
}

impl PCA {
    /// Fits the PCA model to the data and computes the principal components.
    pub fn fit_helper(&mut self, data: &DMatrix<f64>, n_components: usize) {
        let (n_samples, n_features) = data.shape();
        assert!(
            n_components <= n_features,
            "n_components must be <= number of features"
        );

        // Compute the mean of each feature
        self.mean = data.row_mean();

        // Center the data by subtracting the mean
        let centered_data = data - DMatrix::from_rows(&vec![self.mean.clone(); n_samples]);

        // Compute the covariance matrix
        let covariance_matrix =
            &centered_data.transpose() * &centered_data / (n_samples as f64 - 1.0);

        // Perform Singular Value Decomposition
        let svd = SVD::new(covariance_matrix, true, true);

        // Extract the top n_components
        self.components = svd.v_t.unwrap().rows(0, n_components).transpose();
        self.explained_variance = svd
            .singular_values
            .rows(0, n_components)
            .map(|s| s * s / (n_samples as f64 - 1.0));
    }

    /// Transforms the data to the principal component space.
    pub fn transform_helper(&self, data: &DMatrix<f64>) -> DMatrix<f64> {
        let n_samples = data.nrows();
        let centered_data = data - DMatrix::from_rows(&vec![self.mean.clone(); n_samples]);
        &centered_data * &self.components
    }

    /// Fits the PCA model and transforms the data.
    pub fn fit_transform_helper(
        &mut self,
        data: &DMatrix<f64>,
        n_components: usize,
    ) -> DMatrix<f64> {
        self.fit_helper(data, n_components);
        self.transform_helper(data)
    }

    /// Inversely transforms the data back to the original feature space.
    pub fn inverse_transform_helper(&self, data: &DMatrix<f64>) -> DMatrix<f64> {
        let n_samples = data.nrows();
        let reconstructed_data = data * self.components.transpose();
        reconstructed_data + DMatrix::from_rows(&vec![self.mean.clone(); n_samples])
    }

    /// Returns the principal components.
    pub fn components_helper(&self) -> &DMatrix<f64> {
        &self.components
    }

    /// Returns the amount of variance explained by each of the selected components.
    pub fn explained_variance_helper(&self) -> &DVector<f64> {
        &self.explained_variance
    }
}
