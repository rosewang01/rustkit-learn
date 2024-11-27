use nalgebra::DMatrix;
use::std::fmt;

#[derive(Debug)]
pub struct ImputerError {
    column_index: usize,
}

impl fmt::Display for ImputerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Column {} has no non-missing values to compute the mean.", self.column_index)
    }
}

impl std::error::Error for ImputerError {}

#[derive(Debug)]
pub enum ImputationType {
    Mean,
    Constant(f64),
}

pub struct Imputer {
    strategy: ImputationType
}

// Performs imputation on a matrix of Option<f64>. Necessary when importing datasets with null entries (e.g. Python)
// Returns a 
impl Imputer {
    pub fn new(strategy: ImputationType) -> Self {
        Imputer { strategy }
    }

    pub fn fit_transform(&self, data: &DMatrix<Option<f64>>) -> Result<DMatrix<f64>, ImputerError> {
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
