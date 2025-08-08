import pandas as pd
import os
import glob
from pathlib import Path
import numpy as np
import gc
import psutil
from tqdm import tqdm

def combine_csv_files(folder_path, value_column=None, timestamp_column=None, output_filename='combined_data.csv', 
                     batch_size=50, precision=4, use_compression=True, optimize_data_types=True, 
                     remove_zero_columns=False, downsample_factor=1):
    """
    Combine multiple CSV files into one, using the timestamp column as the common key
    and extracting specified value columns from each CSV.
    
    Parameters:
    folder_path (str): Path to the folder containing CSV files
    value_column (str): Name of the column to extract values from (if None, will use second column)
    timestamp_column (str): Name of the timestamp column (if None, will use first column)
    output_filename (str): Name of the output CSV file
    batch_size (int): Number of files to process in each batch (default: 50)
    precision (int): Number of decimal places to round ONLY numeric values (default: 4)
    use_compression (bool): Whether to use gzip compression (default: True)
    optimize_data_types (bool): Whether to optimize data types to reduce memory (default: True)
    remove_zero_columns (bool): Whether to remove columns that are all zeros (default: False)
    downsample_factor (int): Factor to downsample data (1=no downsampling, 2=every 2nd sample, etc.)
    """
    
    # Get all CSV files in the folder
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Check available memory
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    print(f"Available memory: {available_gb:.1f} GB")
    
    if len(csv_files) > 500:
        print(f"⚠️  Processing {len(csv_files)} files - using batch processing")
        batch_size = min(batch_size, 25)  # Smaller batches for large datasets
    
    print(f"Processing in batches of {batch_size} files")
    
    # Determine column names by examining the first file
    first_file = csv_files[0]
    sample_df = pd.read_csv(first_file, nrows=5)
    
    # Always use first column as timestamp and second column as value
    if len(sample_df.columns) < 2:
        print(f"Error: Files must have at least 2 columns. Found only {len(sample_df.columns)} columns.")
        return
    
    # Use column positions: first column (index 0) for timestamp, second column (index 1) for value
    timestamp_column = sample_df.columns[0]
    value_column = sample_df.columns[1]
    
    print(f"\nAuto-detected columns:")
    print(f"  - Timestamp column (1st): '{timestamp_column}'")
    print(f"  - Value column (2nd): '{value_column}'")
    
    # Show a preview of the first file structure
    print(f"\nSample data from {os.path.basename(first_file)}:")
    print(sample_df.head(3).to_string())
    
    print(f"\nPhase 2: Processing {len(csv_files)} files...")
    print(f"Using '{value_column}' as the value column")
    print(f"Using '{timestamp_column}' as the timestamp column")
    
    # Initialize the combined dataframe
    combined_df = None
    
    # Process files in batches to manage memory
    for batch_start in range(0, len(csv_files), batch_size):
        batch_end = min(batch_start + batch_size, len(csv_files))
        batch_files = csv_files[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(csv_files)-1)//batch_size + 1} "
              f"({len(batch_files)} files)")
        
        # Process each file in the batch
        for csv_file in tqdm(batch_files, desc="Processing files"):
            try:
                # Read the CSV file - keep original data types
                df = pd.read_csv(csv_file)
                
                # Get the filename without extension to use as column name
                filename = Path(csv_file).stem
                
                # Use column positions instead of names for robustness
                if len(df.columns) < 2:
                    print(f"    ⚠️  File {filename} has less than 2 columns, skipping...")
                    continue
                
                # Always use first column as timestamp, second as value (by position)
                actual_timestamp_col = df.columns[0]
                actual_value_col = df.columns[1]
                
                # Extract only the required columns
                file_data = df[[actual_timestamp_col, actual_value_col]].copy()
                
                # Rename columns to standard names for consistency
                file_data = file_data.rename(columns={
                    actual_timestamp_col: 'Timestamp',
                    actual_value_col: filename
                })
                
                # Convert timestamp to numeric for proper sorting/merging
                file_data['Timestamp'] = pd.to_numeric(file_data['Timestamp'], errors='coerce')
                
                # Handle value column carefully - preserve original data type and values
                original_values = file_data[filename].copy()
                
                # Try to convert to numeric, but keep original if conversion fails
                try:
                    numeric_values = pd.to_numeric(file_data[filename], errors='coerce')
                    # Only round if values are actually numeric
                    if not numeric_values.isna().all():
                        # Round only the numeric values, keep original for non-numeric
                        mask = ~numeric_values.isna()
                        file_data.loc[mask, filename] = numeric_values[mask].round(precision)
                        # Keep original values for non-numeric entries
                        file_data.loc[~mask, filename] = original_values[~mask]
                        
                        # Optimize data types if requested
                        if optimize_data_types:
                            # Try to use smaller data types
                            numeric_col = file_data[filename]
                            if pd.api.types.is_numeric_dtype(numeric_col):
                                # Check if values can fit in smaller integer types
                                min_val = numeric_col.min()
                                max_val = numeric_col.max()
                                
                                if pd.isna(min_val) or pd.isna(max_val):
                                    pass  # Keep as is if all NaN
                                elif min_val >= 0 and max_val <= 255:
                                    file_data[filename] = file_data[filename].astype('uint8')
                                elif min_val >= -128 and max_val <= 127:
                                    file_data[filename] = file_data[filename].astype('int8')
                                elif min_val >= 0 and max_val <= 65535:
                                    file_data[filename] = file_data[filename].astype('uint16')
                                elif min_val >= -32768 and max_val <= 32767:
                                    file_data[filename] = file_data[filename].astype('int16')
                                elif min_val >= -2147483648 and max_val <= 2147483647:
                                    file_data[filename] = file_data[filename].astype('int32')
                                else:
                                    file_data[filename] = file_data[filename].astype('float32')
                    else:
                        # All values are non-numeric, keep original
                        file_data[filename] = original_values
                except Exception:
                    # If conversion fails completely, keep original values
                    file_data[filename] = original_values
                
                # Remove rows with invalid timestamps
                file_data = file_data.dropna(subset=['Timestamp'])
                
                # Debug: Show sample of data being processed
                non_zero_count = 0
                zero_count = 0
                try:
                    numeric_check = pd.to_numeric(file_data[filename], errors='coerce')
                    zero_count = (numeric_check == 0).sum()
                    non_zero_count = (~numeric_check.isna() & (numeric_check != 0)).sum()
                except Exception:
                    pass
                
                print(f"    ✓ {filename}: {len(file_data)} rows, {zero_count} zeros, {non_zero_count} non-zero values")
                
                # Merge with combined dataframe
                if combined_df is None:
                    combined_df = file_data
                else:
                    combined_df = pd.merge(combined_df, file_data, on='Timestamp', how='outer')
                
                # Clean up memory
                del df, file_data
                gc.collect()
                
            except Exception as e:
                print(f"    ✗ Error processing {os.path.basename(csv_file)}: {str(e)}")
                continue
        
        print(f"    ✓ Batch {batch_start//batch_size + 1} completed")
    
    print(f"\nPhase 3: Finalizing combined data...")
    
    if combined_df is not None:
        # Sort by timestamp to ensure proper order
        combined_df = combined_df.sort_values(by='Timestamp').reset_index(drop=True)
        
        # Create a continuous sample index
        print("  - Creating continuous sample index...")
        
        # Get the min and max timestamps to determine the range
        min_timestamp = combined_df['Timestamp'].min()
        max_timestamp = combined_df['Timestamp'].max()
        
        # Find the most common time interval (sampling rate)
        time_diffs = combined_df['Timestamp'].diff().dropna()
        time_diffs = time_diffs[time_diffs > 0]  # Remove zeros and negative values
        
        if len(time_diffs) > 0:
            # Use the most common time difference as the sampling interval
            sampling_interval = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else time_diffs.min()
        else:
            sampling_interval = 125  # Default based on your example
        
        print(f"  - Timestamp range: {min_timestamp} to {max_timestamp}")
        print(f"  - Detected sampling interval: {sampling_interval}")
        
        # Create a continuous timestamp range
        continuous_timestamps = pd.Series(range(int(min_timestamp), int(max_timestamp) + int(sampling_interval), int(sampling_interval)))
        continuous_df = pd.DataFrame({'Timestamp': continuous_timestamps})
        
        # Merge with the combined data using left join to keep all timestamps
        final_df = pd.merge(continuous_df, combined_df, on='Timestamp', how='left')
        
        # Apply downsampling if requested
        if downsample_factor > 1:
            print(f"  - Downsampling by factor of {downsample_factor}...")
            final_df = final_df.iloc[::downsample_factor].reset_index(drop=True)
        
        # Remove columns that are all zeros, if requested
        if remove_zero_columns:
            print("  - Removing columns with all zeros...")
            parameter_columns = [col for col in final_df.columns if col != 'Timestamp']
            columns_to_remove = []
            
            for col in parameter_columns:
                try:
                    numeric_col = pd.to_numeric(final_df[col], errors='coerce')
                    if not numeric_col.isna().all() and (numeric_col == 0).all():
                        columns_to_remove.append(col)
                except Exception:
                    pass
            
            if columns_to_remove:
                print(f"    Removing {len(columns_to_remove)} zero-only columns: {columns_to_remove[:5]}{'...' if len(columns_to_remove) > 5 else ''}")
                final_df = final_df.drop(columns=columns_to_remove)
        
        # Forward fill missing values (use previous valid value)
        print("  - Forward filling missing values...")
        parameter_columns = [col for col in final_df.columns if col != 'Timestamp']
        
        for col in parameter_columns:
            # Use forward fill 
            final_df[col] = final_df[col].ffill()
        
        # If there are still NaN values at the beginning, backward fill
        for col in parameter_columns:
            final_df[col] = final_df[col].bfill()
        
        # Replace any remaining NaN values with 0 (only if the column was originally numeric)
        for col in parameter_columns:
            if final_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                final_df[col] = final_df[col].fillna(0)
            else:
                # For non-numeric columns, fill with empty string
                final_df[col] = final_df[col].fillna('')
        
        # Add a continuous sample column starting from 0
        final_df.insert(0, 'Sample', range(len(final_df)))
        
        # Reorder columns: Sample, Timestamp, then parameter columns
        column_order = ['Sample', 'Timestamp'] + parameter_columns
        final_df = final_df[column_order]
        
        # Optimize data types for final dataframe if requested
        if optimize_data_types:
            print("  - Optimizing data types to reduce memory usage...")
            for col in parameter_columns:
                if final_df[col].dtype == np.float64:
                    final_df[col] = final_df[col].astype(np.float32)
                elif final_df[col].dtype == np.int64:
                    final_df[col] = final_df[col].astype(np.int32)
        
        # Save the combined CSV (with optional compression)
        output_path = os.path.join(folder_path, output_filename)
        
        # Adjust output filename for compression
        if use_compression and not output_filename.endswith('.gz'):
            if output_filename.endswith('.csv'):
                output_path = output_path[:-4] + '.csv.gz'
            else:
                output_path = output_path + '.gz'
        
        print(f"  - Saving {'with gzip compression' if use_compression else 'without compression'}...")
        
        if use_compression:
            final_df.to_csv(output_path, index=False, compression='gzip')
        else:
            final_df.to_csv(output_path, index=False)
        
        # Calculate file size and show size comparison
        file_size_mb = os.path.getsize(output_path) / (1024**2)
        
        # Estimate uncompressed size if compressed
        if use_compression:
            # Rough estimate: gzip compression typically achieves 3-10x compression
            estimated_uncompressed_mb = file_size_mb * 5  # Conservative estimate
            print(f"\n✓ Combined CSV saved as: {os.path.basename(output_path)}")
            print(f"✓ Compressed file size: {file_size_mb:.1f} MB")
            print(f"✓ Estimated uncompressed size: ~{estimated_uncompressed_mb:.1f} MB")
            print(f"✓ Compression ratio: ~{estimated_uncompressed_mb/file_size_mb:.1f}x smaller")
        else:
            print(f"\n✓ Combined CSV saved as: {os.path.basename(output_path)}")
            print(f"✓ File size: {file_size_mb:.1f} MB")
            
        print(f"✓ Total rows: {len(final_df):,}")
        print(f"✓ Total columns: {len(final_df.columns)}")
        print(f"✓ Sample range: 0 to {len(final_df)-1:,}")
        
        if downsample_factor > 1:
            original_rows = len(final_df) * downsample_factor
            print(f"✓ Downsampled from {original_rows:,} to {len(final_df):,} rows ({100/downsample_factor:.1f}% of original)")
        
        if remove_zero_columns and 'columns_to_remove' in locals():
            print(f"✓ Removed {len(columns_to_remove)} zero-only columns")
        
        # Show a preview of the combined data
        print(f"\nPreview of combined data:")
        print(final_df.head(10).to_string())
        
        # Show statistics about data completeness
        print(f"\nData completeness:")
        for col in parameter_columns:
            if final_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                missing_count = final_df[col].isna().sum()
                zero_count = (final_df[col] == 0).sum()
                non_zero_count = ((final_df[col] != 0) & (~final_df[col].isna())).sum()
                print(f"  - {col}: {non_zero_count} non-zero, {zero_count} zeros, {missing_count} missing")
            else:
                empty_count = (final_df[col] == '').sum()
                non_empty_count = (final_df[col] != '').sum()
                print(f"  - {col}: {non_empty_count} non-empty, {empty_count} empty (text column)")
        
        return output_path
    else:
        print("No data was successfully processed.")
        return None

def main():
    """
    Main function to run the CSV combiner
    """
    print("=== CSV File Combiner ===")
    print("This program combines multiple CSV files into one.")
    print("\n🗜️  File Size Optimization Features:")
    print("  • Gzip compression: 70-90% size reduction")
    print("  • Data type optimization: 20-50% size reduction") 
    print("  • Remove zero-only columns: Variable reduction")
    print("  • Downsampling: Proportional reduction")
    print("  • All optimizations can be combined for maximum effect")
    print()
    
    # Get the current folder path
    current_folder = os.getcwd()
    
    # Ask user for folder path or use current folder
    folder_input = input(f"Enter folder path (press Enter to use current folder: {current_folder}): ").strip()
    
    if folder_input:
        folder_path = folder_input
    else:
        folder_path = current_folder
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Ask for output filename
    output_name = input("Enter output filename (press Enter for 'combined_data.csv'): ").strip()
    if not output_name:
        output_name = 'combined_data.csv'
    
    # Ensure the output filename has .csv extension
    if not output_name.endswith('.csv'):
        output_name += '.csv'
    
    # Ask for optimization settings
    print("\n=== Processing Settings ===")
    batch_size = input("Enter batch size for processing (default: 50): ").strip()
    batch_size = int(batch_size) if batch_size.isdigit() else 50
    
    precision = input("Enter decimal precision for NUMERIC values only (default: 4): ").strip()
    precision = int(precision) if precision.isdigit() else 4
    
    use_compression = input("Use gzip compression for output file? (y/n, default: y): ").strip().lower()
    use_compression = use_compression == 'y' if use_compression else True
    
    optimize_data_types = input("Optimize data types to reduce memory usage? (y/n, default: y): ").strip().lower()
    optimize_data_types = optimize_data_types == 'y' if optimize_data_types else True
    
    remove_zero_columns = input("Remove columns that are all zeros? (y/n, default: n): ").strip().lower()
    remove_zero_columns = remove_zero_columns == 'y' if remove_zero_columns else False
    
    downsample_factor = input("Downsample data by a factor (default: 1, no downsampling): ").strip()
    downsample_factor = int(downsample_factor) if downsample_factor.isdigit() else 1
    
    print(f"\nSettings:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Precision: {precision} decimal places (numeric values only)")
    print(f"  - Output: {'Gzip compressed CSV' if use_compression else 'Uncompressed CSV'}")
    print(f"  - Optimize data types: {'Yes' if optimize_data_types else 'No'}")
    print(f"  - Remove zero columns: {'Yes' if remove_zero_columns else 'No'}")
    print(f"  - Downsample factor: {downsample_factor}")
    
    # Run the combination process
    try:
        result = combine_csv_files(folder_path, output_filename=output_name, 
                                 batch_size=batch_size, precision=precision,
                                 use_compression=use_compression, 
                                 optimize_data_types=optimize_data_types,
                                 remove_zero_columns=remove_zero_columns,
                                 downsample_factor=downsample_factor)
        if result:
            print(f"\n🎉 Success! Combined file created at: {result}")
        else:
            print("\n❌ Failed to create combined file.")
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()