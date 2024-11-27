use pyo3::prelude::*;

pub mod converters;
mod preprocessing;
use converters::*;
use preprocessing::standard_scaler::StandardScaler;

#[pymodule]
fn rust_final_proj(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add StandardScaler as a Python class
    m.add_class::<StandardScaler>()?;

    Ok(())
}
