
# CSV File Combiner & Optimizer

This project provides a powerful Python script (`main.py`) for combining and optimizing large numbers of CSV files. It is designed for efficient data aggregation, memory management, and file size reduction, making it ideal for processing time-series or sensor data exported as CSVs.

## What Does It Do?

The script interactively guides you through combining multiple CSV files from a folder into a single, well-structured CSV file. It automatically detects timestamp and value columns, merges data by timestamp, and offers advanced features for optimizing the output file:

- **Batch Processing:** Handles thousands of files in memory-efficient batches.
- **Gzip Compression:** Reduces output file size by 70-90%.
- **Data Type Optimization:** Shrinks memory usage by converting columns to the smallest suitable types.
- **Remove Zero-Only Columns:** Optionally drops columns that contain only zeros.
- **Downsampling:** Reduces the number of rows by keeping every Nth sample.
- **Forward/Backward Fill:** Fills missing values for continuous time series.
- **Customizable Settings:** All options are set interactively at runtime.

## Features

- **Automatic column detection** (timestamp/value)
- **Batch processing** for large datasets
- **Memory usage reporting**
- **Progress bars** for file processing
- **Flexible output options** (compression, precision, etc.)
- **Data completeness statistics**

## Usage

1. **Requirements:**
    - Python 3.x
    - Packages: `pandas`, `numpy`, `psutil`, `tqdm`
    - Install dependencies:
       ```powershell
       pip install pandas numpy psutil tqdm
       ```

2. **How to Run:**
    - Place all your CSV files in a folder.
    - Open a terminal in the project directory.
    - Run the script:
       ```powershell
       python main.py
       ```
    - Follow the interactive prompts to select the folder, output filename, and processing options.

3. **Output:**
    - The script creates a combined CSV (optionally compressed) with a continuous sample index, merged columns, and optimized data types.

## Example

```
=== CSV File Combiner ===
This program combines multiple CSV files into one.

🗜️  File Size Optimization Features:
   • Gzip compression: 70-90% size reduction
   • Data type optimization: 20-50% size reduction
   • Remove zero-only columns: Variable reduction
   • Downsampling: Proportional reduction
   • All optimizations can be combined for maximum effect

Enter folder path (press Enter to use current folder: ...):
Enter output filename (press Enter for 'combined_data.csv'):
... (other prompts)
```

## Project Structure

- `main.py` — Main script for combining and optimizing CSV files.

## License

This project is provided as-is. Add a license if needed.
# combine-flight-csvs
