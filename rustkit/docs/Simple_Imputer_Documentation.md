
# Rust `Imputer` Class Documentation

This documentation describes the `Imputer` class, a utility inspired by Scikit-learn for performing missing data imputation on a dataset represented as a matrix. The `Imputer` provides strategies to handle missing values, such as replacing them with the mean of a column or a constant value.

---

## **Structs**

### `Imputer`

The main struct for handling imputation tasks. It allows imputing missing values in datasets with configurable strategies.

#### **Fields**

- `strategy`: `ImputationType`  
   The imputation strategy to use (e.g., mean imputation or constant value imputation).

---

### `ImputerError`

An error type representing a failure in the imputation process, specifically when a column lacks non-missing values required for mean imputation.

#### **Fields**

- `column_index`: `usize`  
   The index of the column where the error occurred.

#### **Traits Implemented**

- `fmt::Display`  
  Provides a user-friendly error message indicating the column with the issue.
- `std::error::Error`  
  Implements the `Error` trait for compatibility with error-handling idioms.

---

## **Enums**

### `ImputationType`

Represents the strategy to use for imputing missing values.

#### **Variants**

- `Mean`  
  Impute missing values with the mean of the non-missing values in the column.

- `Constant(f64)`  
  Impute missing values with a specified constant value.

---

## **Methods**

### `Imputer::new`

Creates a new `Imputer` instance with the specified imputation strategy.

#### **Signature**

```rust
pub fn new(strategy: ImputationType) -> Self
```

#### **Parameters**

- `strategy`: `ImputationType`  
   The strategy to use for imputing missing values.

#### **Returns**

- An `Imputer` instance configured with the given strategy.

---

### `Imputer::fit_transform`

Performs the imputation on a dataset represented as a matrix of optional floating-point numbers (`DMatrix<Option<f64>>`).

#### **Signature**

```rust
pub fn fit_transform(&self, data: &DMatrix<Option<f64>>) -> Result<DMatrix<f64>, ImputerError>
```

#### **Parameters**

- `data`: `&DMatrix<Option<f64>>`  
   A reference to the input dataset with potential missing values (`Option<f64>`).

#### **Returns**

- `Ok(DMatrix<f64>)`  
   A matrix where all missing values have been imputed based on the chosen strategy.
- `Err(ImputerError)`  
   If mean imputation is selected and a column has no non-missing values, the method returns an `ImputerError`.

---

## **Usage Example**

```rust
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example dataset with missing values (None)
    let data = DMatrix::from_vec(
        3,
        3,
        vec![
            Some(1.0), None, Some(3.0),
            None, None, Some(6.0),
            Some(7.0), Some(8.0), None,
        ],
    );

    // Create an imputer with the mean strategy
    let imputer = Imputer::new(ImputationType::Mean);

    // Perform the imputation
    let transformed = imputer.fit_transform(&data)?;

    println!("Imputed Data:\n{}", transformed);

    Ok(())
}
```

---

## **Error Handling**

When using the `fit_transform` method with the `Mean` strategy, ensure that each column has at least one non-missing value. If a column is entirely missing, the method will return an `ImputerError` with the problematic column index.

---

## **Dependencies**

This implementation relies on the following external crates:

- `nalgebra`  
  Provides the `DMatrix` type used for matrix operations.

To include these dependencies, add the following to your `Cargo.toml`:

```toml
[dependencies]
nalgebra = "0.31"
```

---

## **Notes**

- The `Imputer` currently supports only mean and constant value imputation.
- For large datasets, consider optimizing the implementation for performance.

---

Happy coding! 🎉
