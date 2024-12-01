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
