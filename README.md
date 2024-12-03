# rustkit

---

## Overview

`rustkit` is a data science library written in Rust, callable from Python, and inspired by `Scikit-Learn`. The underlying Rust implementation relies on the `nalegbra` crate for fast linear algebra and matrix/vector data structures. Like Scikit-Learn, methods are defined through classes (structs) and tend to follow a fit-transform or fit-predict approach for the model (for our supervised, unsupervised, and preprocessing models). Additionally we provide some structs to define testin methods (such as R^2) score etc.

This project includes Python bindings using `maturin` and `PyO3` to use these methods and classes as a library in Python, called `rustkit`. To do so, we implemented converter functions that converted `numpy` matrices and vectors into `nalgebra` matrices and vectors, handling generic types and null values. More information can be found below on building the library for Python. 

For now, the methods only accept floats (represented as `f64` in Rust). So far, we have implemented the classes below, grouped by type:

- Preprocessing:
    - Scaler
    - Imputer
- Supervised
    - Ridge Regression
    - With the following Regression Metrics:
        - $R^2$
        - MSE
- Unsupervised
    - KMeans
    - PCA

After implementing the methods in Rust, we created Python bindings using `maturin` and `PyO3` to use these methods and classes as a library in Python, called `rustkit`. To do so, we implemented converter functions that converted `numpy` matrices and vectors into `nalgebra` matrices and vectors, handling generic types and null values.

***Note***
Numpy matrices and pandas dataframes in Python tend to handle `None` or `NaN` entries. In Rust, while we can have null entries by storing our data as a `Option<f64>` matrix, most matrix operations are not implementable on Optional values. Thus all of our methods expect ***non-null*** entries, with the exception of `SimpleImputer` which provides imputation methods to ensure that input data is completely non-null.


## Project Structure

This repo contains a folder defining the rustkit [library](rustkit/), as well as a folder containing unit-tests (which compare our output with Sk-learn) and benchmarking in [Python](python/).

### ***rustkit/***
- `src` contains all of the of the Rust code needed. The core algorithms are organized by type into modules (e.g. preprocessing, supervised, etc.). Documentation for each class can be found below
- `src/main.rs` prints an example use of all of these algorithms directly in Rust (use `cargo run` from the [rustkit/](rustkit/) folder).
- ***add exmplanations for lib, converter, and benchmarking***


### ***python/***
- ***briefly give an overview of this folder****.




## Building the Library

- Make sure that you `pip install maturin` in order to be able to build the package.
- To build the package, run `maturin develop` from the `rustkit` directory
  - use `maturin develop --release` if you want the build to be optimized
  - This command compiles the local rust code into a Python module and installs it into your local Python environment
  - If this doesn't work / you get some sort of warning, check to see if you have the `rustkit/rustkit/` folder exists. If not, create it.
- To use in a python module:
  - Within the `rustkit/rustkit/` folder, create `__init__.py`
  - Put the following in `__init__.py`:
  ```
  from .rustkit import *
  from .rustkit import __all__
  ```
  - Within your python module, import and use rustkit as follows: `from rustkit import StandardScaler` or `import rustkit`


## Documentation

The underlying data science algorithms are implemented in Rust and rely heavily on the `nalgebra` crate. Below is the documentation for the classes currently available in this project. Each documentation page separates the methods into the external methods (those intended for Python integration) and internal methods (the Rust code to actually run the algorithms / update parameters). Classes are grouped by their type.

### **Preprocessing**

- [SimpleImputer](docs/Simple_Imputer_Documentation.md)
- [StandardScaler](docs/Standard_Scaler_Documentation.md)

### **Unsupervised**

- [KMeans](docs/KMeans_Documentation.MD)
- [PCA](docs/PCA_Documentation.MD)

### **Supervised**

- [RidgeRegression](docs/Ridge_Regression_Documentation.md)

### **Testing**

- [R2Score](docs/R2_Score_Documentation.md)
- [MSE](docs/MSE_Score_Documentation.md)