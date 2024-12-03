# rust-final-proj

---

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