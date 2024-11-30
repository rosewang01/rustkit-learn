use crate::converters::{
    python_to_rust_dynamic_matrix, python_to_rust_dynamic_vector, rust_to_python_dynamic_vector,
    rust_to_python_opt_float,
};
use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyFloat;

#[pyclass]
pub struct RidgeRegression {
    weights: DVector<f64>,
    intercept: Option<f64>, // Optional because it may not be used
    regularization: f64,
    with_bias: bool,
}

#[pymethods]
impl RidgeRegression {
    /// Creates a new RidgeRegression instance.
    /// Set `regularization` to 0.0 for standard linear regression.
    /// The `with_bias` parameter specifies whether to include a bias term (default is true).
    /// We use LU decomposition and solve, rather than explicitly computing a (computationally expensive) matrix inverse
    /// IMPORTANT: Ridge regression should only be used on normalized data. If the data is not normalized, set regularization = 0.

    #[new]
    pub fn new(regularization: f64, with_bias: bool) -> Self {
        RidgeRegression {
            weights: DVector::zeros(0),
            intercept: if with_bias { Some(0.0) } else { None },
            regularization,
            with_bias,
        }
    }

    #[getter]
    pub fn weights(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        rust_to_python_dynamic_vector(py, self.weights.clone())
    }

    #[getter]
    pub fn intercept(&self, py: Python) -> PyResult<Py<PyFloat>> {
        rust_to_python_opt_float(py, self.intercept)
    }

    #[setter]
    pub fn set_regularization(&mut self, regularization: f64) -> PyResult<()> {
        self.regularization = regularization;
        Ok(())
    }

    #[setter]
    pub fn set_with_bias(&mut self, with_bias: bool) -> PyResult<()> {
        self.with_bias = with_bias;
        if with_bias {
            self.intercept = Some(0.0);
        } else {
            self.intercept = None;
        }
        Ok(())
    }

    #[setter]
    pub fn set_weights(&mut self, weights: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.weights = python_to_rust_dynamic_vector(&weights);
        Ok(())
    }

    #[setter]
    pub fn set_intercept(&mut self, intercept: Option<f64>) -> PyResult<()> {
        self.intercept = intercept;
        Ok(())
    }

    /// Fits the Ridge Regression model to the data.
    pub fn fit(
        &mut self,
        data: PyReadonlyArray2<f64>,
        target: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let x = python_to_rust_dynamic_matrix(&data);
        let y = python_to_rust_dynamic_vector(&target);
        self.fit_helper(&x, &y);
        Ok(())
    }

    /// Predicts target values for the given input data.
    pub fn predict(&self, py: Python, data: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let x = python_to_rust_dynamic_matrix(&data);
        let predictions = self.predict_helper(&x);
        rust_to_python_dynamic_vector(py, predictions)
    }
}

impl RidgeRegression {
    /// Fits the Ridge Regression model to the data.
    pub fn fit_helper(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        assert_eq!(y.len(), n_samples, "Mismatched input dimensions.");

        if self.with_bias {
            // Add a column of ones to X for the intercept term
            let x_with_bias = {
                let mut extended = DMatrix::zeros(n_samples, n_features + 1);
                extended.index_mut((.., ..n_features)).copy_from(x);
                extended.column_mut(n_features).fill(1.0);
                extended
            };

            // Compute the regularization matrix
            let regularization_matrix = {
                let mut reg_matrix = DMatrix::identity(n_features + 1, n_features + 1);
                reg_matrix[(n_features, n_features)] = 0.0; // Don't regularize the intercept
                reg_matrix * self.regularization
            };

            // Solve for weights and intercept: (X'X + λI)^-1 X'Y
            let xt = x_with_bias.transpose();
            let xtx = &xt * &x_with_bias;
            let xtx_reg = &xtx + regularization_matrix;
            let xty = xt * y;

            let solution = xtx_reg.lu().solve(&xty).expect("Matrix inversion failed.");

            // Extract weights and intercept
            self.weights = solution.rows(0, n_features).into();
            self.intercept = Some(solution[n_features]);
        } else {
            // Compute the regularization matrix for weights only
            let regularization_matrix =
                DMatrix::identity(n_features, n_features) * self.regularization;

            // Solve for weights: (X'X + λI)^-1 X'Y
            let xt = x.transpose();
            let xtx = &xt * x;
            let xtx_reg = &xtx + regularization_matrix;
            let xty = xt * y;

            self.weights = xtx_reg.lu().solve(&xty).expect("Matrix inversion failed.");
            self.intercept = None;
        }
    }

    /// Predicts target values for the given input data.
    pub fn predict_helper(&self, x: &DMatrix<f64>) -> DVector<f64> {
        let predictions = if self.with_bias {
            let intercept = self
                .intercept
                .expect("Bias term is not available but required for prediction.");
            x * &self.weights + DVector::from_element(x.nrows(), intercept)
        } else {
            x * &self.weights
        };
        predictions
    }

    /// Returns the model weights (coefficients).
    pub fn weights_helper(&self) -> &DVector<f64> {
        &self.weights
    }

    /// Returns the model intercept (if available).
    pub fn intercept_helper(&self) -> Option<f64> {
        self.intercept
    }
}
