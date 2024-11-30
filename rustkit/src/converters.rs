use nalgebra::{DMatrix, DVector};
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyFloat;

pub fn python_to_rust_opt_float(value: f64) -> Option<f64> {
    if value.is_nan() {
        None
    } else {
        Some(value)
    }
}

pub fn python_to_rust_dynamic_vector<
    T: Clone + Copy + numpy::Element + std::fmt::Debug + std::cmp::PartialEq + 'static,
>(
    array: &PyReadonlyArray1<T>,
) -> DVector<T> {
    let vector = array.as_array();
    let elements = vector.iter().cloned().collect::<Vec<_>>();
    DVector::from_row_slice(&elements)
}

pub fn python_to_rust_dynamic_matrix<
    T: Clone + Copy + numpy::Element + std::fmt::Debug + std::cmp::PartialEq + 'static,
>(
    array: &PyReadonlyArray2<T>,
) -> DMatrix<T> {
    let matrix = array.as_array();
    let shape = matrix.shape();
    let rows = shape[0];
    let cols = shape[1];
    let elements = matrix.iter().cloned().collect::<Vec<_>>();
    DMatrix::from_row_slice(rows, cols, &elements)
}

pub fn python_to_rust_opt_dynamic_vector(array: &PyReadonlyArray1<f64>) -> DVector<Option<f64>> {
    // Convert a NumPy array to nalgebra::DVector
    let vector = array.as_array();
    let elements = vector.iter().cloned().collect::<Vec<_>>();
    DVector::from_iterator(
        elements.len(),
        elements.into_iter().map(python_to_rust_opt_float),
    )
}

pub fn python_to_rust_opt_dynamic_matrix(array: &PyReadonlyArray2<f64>) -> DMatrix<Option<f64>> {
    // Convert a NumPy array to nalgebra::DMatrix
    let matrix = array.as_array();
    let shape = matrix.shape();
    let rows = shape[0];
    let cols = shape[1];
    let elements = matrix.iter().cloned().collect::<Vec<_>>();
    DMatrix::from_iterator(
        rows,
        cols,
        elements.into_iter().map(python_to_rust_opt_float),
    )
}

pub fn rust_to_python_opt_float(py: Python, value: Option<f64>) -> PyResult<Py<PyFloat>> {
    match value {
        Some(v) => Ok(PyFloat::new(py, v).into()),
        None => Ok(PyFloat::new(py, f64::NAN).into()),
    }
}

pub fn rust_to_python_dynamic_vector<T: ToPyObject + numpy::Element>(
    py: Python,
    vector: DVector<T>,
) -> PyResult<Py<PyArray1<T>>> {
    let array = Array1::from_vec(vector.data.into());
    Ok(PyArray1::from_array(py, &array).into())
}

pub fn rust_to_python_dynamic_matrix<
    T: ToPyObject + Clone + std::fmt::Debug + std::cmp::PartialEq + numpy::Element + 'static,
>(
    py: Python,
    matrix: DMatrix<T>,
) -> PyResult<Py<PyArray2<T>>> {
    let shape = matrix.shape();
    let rows = shape.0;
    let cols = shape.1;
    let transposed_matrix = matrix.transpose();
    let array = Array2::from_shape_vec((cols, rows), transposed_matrix.data.into());
    match array {
        Ok(arr) => Ok(PyArray2::from_array(py, &arr).into()),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Array creation failed: {}",
            e
        ))),
    }
}

pub fn rust_to_python_opt_dynamic_vector(
    py: Python,
    vector: DVector<Option<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    // Convert a nalgebra::DVector (constructed from elements) to a NumPy array
    let array = Array1::from_vec(
        vector
            .data
            .as_slice()
            .iter()
            .map(|v| match v {
                Some(val) => *val,
                None => f64::NAN,
            })
            .collect(),
    );
    Ok(PyArray1::from_array(py, &array).into())
}

pub fn rust_to_python_opt_dynamic_matrix(
    py: Python,
    matrix: DMatrix<Option<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    // Convert a nalgebra::DMatrix (constructed from rows, cols, and elements) to a NumPy array
    let shape = matrix.shape();
    let rows = shape.0;
    let cols = shape.1;

    let transposed_matrix = matrix.transpose();

    let array_data: Vec<f64> = transposed_matrix
        .data
        .as_slice()
        .iter()
        .map(|v| match v {
            Some(val) => *val,
            None => f64::NAN,
        })
        .collect();

    let array = Array2::from_shape_vec((cols, rows), array_data).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Array creation failed: {}", e))
    })?;
    let numpy_array = PyArray2::from_array(py, &array);

    Ok(numpy_array.to_owned())
}

#[pyfunction]
pub fn converter_vector_test(
    py: Python,
    vector: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let rust_vector = python_to_rust_dynamic_vector(&vector);
    rust_to_python_dynamic_vector(py, rust_vector)
}

#[pyfunction]
pub fn converter_matrix_test(
    py: Python,
    matrix: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let rust_matrix = python_to_rust_dynamic_matrix(&matrix);
    rust_to_python_dynamic_matrix(py, rust_matrix)
}
