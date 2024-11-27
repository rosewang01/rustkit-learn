use crate::python_to_rust_dynamic_matrix;
use crate::rust_to_python_dynamic_matrix;
use nalgebra::{DMatrix, DVector};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass]
pub struct StandardScaler {
    means: Option<DVector<f64>>,
    std_devs: Option<DVector<f64>>,
}

//TODO: Add a flag to decide whether feature std is calculated
// by dividing by n or n-1
#[pymethods]
impl StandardScaler {
    /// Creates a new `StandardScaler` instance.
    #[new]
    pub fn new() -> Self {
        StandardScaler {
            means: None,
            std_devs: None,
        }
    }

    /// Fits the scaler to the data by calculating the means and standard deviations for each column.
    pub fn fit(&mut self, data: &PyReadonlyArray2<f64>) -> PyResult<()> {
        let data: DMatrix<f64> = python_to_rust_dynamic_matrix(data);
        let (n_rows, n_cols) = data.shape();

        let mut means = DVector::zeros(n_cols);
        let mut std_devs = DVector::zeros(n_cols);

        for col in 0..n_cols {
            let column = data.column(col);
            let sum: f64 = column.iter().sum();
            let mean = sum / n_rows as f64;

            means[col] = mean;

            let variance: f64 =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_rows) as f64;
            std_devs[col] = variance.sqrt();
        }

        self.means = Some(means);
        self.std_devs = Some(std_devs);
        Ok(())
    }

    /// Transforms the input data using the fitted means and standard deviations.
    pub fn transform(
        &self,
        py: Python,
        data: &PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let data: DMatrix<f64> = python_to_rust_dynamic_matrix(data);
        let means = self
            .means
            .as_ref()
            .expect("Scaler has not been fitted yet.");
        let std_devs = self
            .std_devs
            .as_ref()
            .expect("Scaler has not been fitted yet.");

        let (n_rows, n_cols) = data.shape();
        let mut scaled_data = DMatrix::zeros(n_rows, n_cols);

        for col in 0..n_cols {
            let mean = means[col];
            let std_dev = std_devs[col];

            for row in 0..n_rows {
                scaled_data[(row, col)] = (data[(row, col)] - mean) / std_dev;
            }
        }

        rust_to_python_dynamic_matrix(py, n_rows, n_cols, scaled_data.as_slice().to_vec())
    }

    /// Fits the scaler and then transforms the input data.
    pub fn fit_transform(
        &mut self,
        py: Python,
        data: &PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        self.fit(data)?;
        self.transform(py, data)
    }

    pub fn inverse_transform(
        &self,
        py: Python,
        scaled_data: &PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        // Convert the NumPy array to a nalgebra::DMatrix
        let scaled_data: DMatrix<f64> = python_to_rust_dynamic_matrix(scaled_data);
        let means = self
            .means
            .as_ref()
            .expect("Scaler has not been fitted yet.");
        let std_devs = self
            .std_devs
            .as_ref()
            .expect("Scaler has not been fitted yet.");

        let (n_rows, n_cols) = scaled_data.shape();
        let mut original_data = DMatrix::zeros(n_rows, n_cols);

        for col in 0..n_cols {
            let mean = means[col];
            let std_dev = std_devs[col];

            for row in 0..n_rows {
                original_data[(row, col)] = scaled_data[(row, col)] * std_dev + mean;
            }
        }

        rust_to_python_dynamic_matrix(py, n_rows, n_cols, original_data.as_slice().to_vec())
    }
}
