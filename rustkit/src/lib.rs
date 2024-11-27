use pyo3::prelude::*;

mod preprocessing;
use preprocessing::standard_scaler::StandardScaler;

pub mod converters;
use converters::{converter_matrix_test, converter_vector_test};

#[pymodule]
fn rustkit(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<StandardScaler>()?;
    m.add_function(wrap_pyfunction!(converter_vector_test, m)?)?;
    m.add_function(wrap_pyfunction!(converter_matrix_test, m)?)?;

    Ok(())
}
