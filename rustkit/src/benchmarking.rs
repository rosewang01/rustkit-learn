use std::fs::OpenOptions;
use std::io::{self, Write};
use std::time::Instant;

pub fn log_function_time<F, T>(
    mut func: F,
    func_name: &str,
    input_rows: usize,
    input_cols: usize,
) -> io::Result<T>
where
    F: FnMut() -> T,
{
    let start = Instant::now();
    let resp = func();
    let duration = start.elapsed();

    let runtime = duration.as_secs_f64();

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("timing_log.csv")?;

    writeln!(
        file,
        "{},{},{},{}",
        func_name, input_rows, input_cols, runtime
    )?;

    Ok(resp)
}
