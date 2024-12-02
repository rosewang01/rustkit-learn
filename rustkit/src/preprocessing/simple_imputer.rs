use crate::benchmarking::log_function_time;
use crate::converters::{python_to_rust_opt_dynamic_matrix, rust_to_python_dynamic_matrix};
use nalgebra::DMatrix;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Python;
use std::collections::HashMap;
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
    impute_values: Option<HashMap<usize, f64>>,
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
                impute_values: None,
            },
            "constant" => Imputer {
                strategy: ImputationType::Constant(value.unwrap()),
                impute_values: None,
            },
            _ => panic!("Invalid strategy"),
        }
    }

    pub fn fit(&mut self, data: PyReadonlyArray2<f64>) -> PyResult<()> {
        let rust_data = python_to_rust_opt_dynamic_matrix(&data);
        let result = log_function_time(
            || self.fit_helper(&rust_data),
            "Imputer::fit",
            rust_data.shape().0,
            rust_data.shape().1,
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
        let rust_data = python_to_rust_opt_dynamic_matrix(&data);
        let shape = rust_data.shape();
        let transformed_data = log_function_time(
            || self.transform_helper(&rust_data),
            "Imputer::transform",
            shape.0,
            shape.1,
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

    pub fn fit_transform(
        &mut self,
        py: Python,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let rust_data = python_to_rust_opt_dynamic_matrix(&data);
        let shape = rust_data.shape();
        let transformed_data = log_function_time(
            || self.fit_transform_helper(&rust_data),
            "Imputer::fit_transform",
            shape.0,
            shape.1,
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
    pub fn fit_transform_helper(
        &mut self,
        data: &DMatrix<Option<f64>>,
    ) -> Result<DMatrix<f64>, ImputerError> {
        // Fit the imputer to compute imputation values
        self.fit_helper(data)?;

        // Transform the data using the computed imputation values
        self.transform_helper(data)
    }

    /// Fits the imputer to the data, computing imputation values for each column.
    pub fn fit_helper(&mut self, data: &DMatrix<Option<f64>>) -> Result<(), ImputerError> {
        let (_, ncols) = data.shape();
        let mut impute_values = HashMap::new();

        for j in 0..ncols {
            let column = data.column(j);

            // Compute the imputation value based on the strategy
            let impute_value = match &self.strategy {
                ImputationType::Mean => {
                    let mut non_missing_values = Vec::new();

                    for i in 0..data.nrows() {
                        if let Some(value) = column[i] {
                            non_missing_values.push(value);
                        }
                    }

                    if non_missing_values.is_empty() {
                        return Err(ImputerError { column_index: j }); // Error if column is all None
                    }

                    let sum: f64 = non_missing_values.iter().sum();
                    sum / non_missing_values.len() as f64
                }
                ImputationType::Constant(val) => *val,
            };

            impute_values.insert(j, impute_value);
        }

        self.impute_values = Some(impute_values);
        Ok(())
    }

    /// Transforms the data using the computed imputation values.
    pub fn transform_helper(
        &self,
        data: &DMatrix<Option<f64>>,
    ) -> Result<DMatrix<f64>, ImputerError> {
        let (nrows, ncols) = data.shape();
        let mut result = DMatrix::zeros(nrows, ncols);

        if self.impute_values.is_none() {
            return Err(ImputerError {
                column_index: usize::MAX, // Indicate fitting wasn't done
            });
        }

        let impute_values = self.impute_values.as_ref().unwrap();

        for j in 0..ncols {
            let column = data.column(j);

            let impute_value = impute_values
                .get(&j)
                .ok_or(ImputerError { column_index: j })?;

            for i in 0..nrows {
                result[(i, j)] = column[i].unwrap_or(*impute_value);
            }
        }

        Ok(result)
    }
}
