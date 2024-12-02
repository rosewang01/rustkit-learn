use std::fs::OpenOptions;
use std::io::{self, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub fn log_function_time<F, T>(mut func: F, func_name: &str) -> io::Result<T>
where
    F: FnMut() -> T,
{
    let start_time = SystemTime::now();

    let start = Instant::now();
    let resp = func();
    let duration = start.elapsed();

    let timestamp = match start_time.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs(),
        Err(_) => 0,
    };

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("timing_log.csv")?;

    writeln!(file, "{},{},{:?}", func_name, timestamp, duration)?;

    Ok(resp)
}
