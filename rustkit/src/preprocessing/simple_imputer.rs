use crate::benchmarking::log_function_time;
use crate::converters::{python_to_rust_opt_dynamic_matrix, rust_to_python_dynamic_matrix};
use nalgebra::DMatrix;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Python;
use std::fmt;

#[derive(Debug)]
pub struct ImputerError {
    column_index: usize,
}

impl fmt::Display for ImputerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Column {} has no non-missing values to compute the mean.",
            self.column_index
        )
    }
}

impl std::error::Error for ImputerError {}

#[derive(Debug, Clone)]
pub enum ImputationType {
    Mean,
    Constant(f64),
}

#[pyclass]
pub struct Imputer {
    strategy: ImputationType,
}

// Performs imputation on a matrix of Option<f64>. Necessary when importing datasets with null entries (e.g. Python)
// Returns a
#[pymethods]
impl Imputer {
    #[new]
    pub fn new(strategy: &str, value: Option<f64>) -> Self {
        match strategy {
            "mean" => Imputer {
                strategy: ImputationType::Mean,
            },
            "constant" => Imputer {
                strategy: ImputationType::Constant(value.unwrap()),
            },
            _ => panic!("Invalid strategy"),
        }
    }

    pub fn fit_transform(
        &self,
        py: Python,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let rust_data = python_to_rust_opt_dynamic_matrix(&data);
        let transformed_data = log_function_time(
            || self.fit_transform_helper(&rust_data),
            "Imputer::fit_transform",
        )
        .unwrap();
        match transformed_data {
            Ok(data) => rust_to_python_dynamic_matrix(py, data),
            Err(e) => Err(PyValueError::new_err(format!(
                "Column {} has no non-missing values to compute the mean.",
                e.column_index,
            ))),
        }
    }
}

impl Imputer {
    // pub fn new(strategy: ImputationType) -> Self {
    //     Imputer { strategy }
    // }
    pub fn fit_transform_helper(
        &self,
        data: &DMatrix<Option<f64>>,
    ) -> Result<DMatrix<f64>, ImputerError> {
        let (nrows, ncols) = data.shape();
        let mut result = DMatrix::zeros(nrows, ncols);

        for j in 0..ncols {
            let column = data.column(j);

            // Get the imputation value, either a column mean or a constant
            let impute_value = match &self.strategy {
                ImputationType::Mean => {
                    let mut non_missing_values = Vec::new();

                    for i in 0..nrows {
                        if let Some(value) = column[i] {
                            non_missing_values.push(value);
                        }
                    }

                    if non_missing_values.is_empty() {
                        return Err(ImputerError { column_index: j }); // Can't perform mean imputation if a column is all None, so return an error
                    }
                    let sum: f64 = non_missing_values.iter().sum();
                    sum / non_missing_values.len() as f64
                }
                ImputationType::Constant(val) => *val,
            };

            for i in 0..nrows {
                result[(i, j)] = column[i].unwrap_or(impute_value);
            }
        }

        Ok(result)
    }
}
