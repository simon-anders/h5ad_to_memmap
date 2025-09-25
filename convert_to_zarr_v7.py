#!/usr/bin/env python3
"""
Convert h5ad file with CSR X matrix to zarr with CSC format.
Optimized version with persistent file handles and larger chunks.
"""

import sys
import os
import h5py
import zarr
import numpy as np
from pathlib import Path
from numba import njit

def check_h5ad_format(h5ad_path):
    """Check that h5ad file has CSR format X matrix and get dimensions."""
    with h5py.File(h5ad_path, 'r') as f:
        if 'X' not in f:
            raise ValueError("No X matrix found in h5ad file")
        
        x_group = f['X']
        
        # Check if it's sparse CSR format
        required_keys = ['data', 'indices', 'indptr']
        if not all(key in x_group for key in required_keys):
            raise ValueError(f"X matrix is not in CSR format. Found keys: {list(x_group.keys())}")
        
        # Get dimensions
        if 'shape' in x_group.attrs:
            shape = tuple(x_group.attrs['shape'])
        else:
            # Calculate shape from indptr and indices
            n_rows = len(x_group['indptr']) - 1
            n_cols = int(np.max(x_group['indices'][:])) + 1 if len(x_group['indices']) > 0 else 0
            shape = (n_rows, n_cols)
        
        nnz = len(x_group['data'])
        data_dtype = x_group['data'].dtype
        
        print(f"✓ Found CSR matrix with shape {shape}, {nnz:,} non-zeros, dtype {data_dtype}")
        return shape, nnz, data_dtype

@njit
def count_and_collect_target_entries(data_chunk, indices_chunk, indptr_chunk, 
                                   col_start, col_end, row_start,
                                   out_rows, out_cols, out_values):
    """Combined count and collect in single pass - much faster than separate passes."""
    write_idx = 0
    
    for row_offset in range(len(indptr_chunk) - 1):
        row_idx = row_start + row_offset
        start = indptr_chunk[row_offset]
        end = indptr_chunk[row_offset + 1]
        
        for idx in range(start, end):
            col_idx = indices_chunk[idx]
            if col_start <= col_idx < col_end:
                if write_idx < len(out_rows):  # Safety check
                    out_rows[write_idx] = row_idx
                    out_cols[write_idx] = col_idx
                    out_values[write_idx] = data_chunk[idx]
                    write_idx += 1
    
    return write_idx

def collect_columns_optimized(h5ad_path, col_start, col_end, n_rows, row_chunk_size=50000):
    """Optimized column collection with single file handle and larger chunks."""
    print(f"  Collecting columns {col_start:,} to {col_end-1:,}...")
    
    # Estimate max entries per chunk (conservative estimate)
    cols_in_range = col_end - col_start
    max_entries_per_chunk = row_chunk_size * cols_in_range // 10  # Assume ~10% sparsity per column range
    
    all_rows_list = []
    all_cols_list = []
    all_values_list = []
    
    total_chunks = (n_rows + row_chunk_size - 1) // row_chunk_size
    
    # Keep file open for entire pass
    with h5py.File(h5ad_path, 'r') as f:
        x_group = f['X']
        
        for chunk_idx, row_start in enumerate(range(0, n_rows, row_chunk_size)):
            if chunk_idx % 1 == 0:  # Progress every 10 chunks
                progress = (chunk_idx / total_chunks) * 100
                total_collected = sum(len(arr) for arr in all_rows_list)
                print(f"    Progress: {chunk_idx}/{total_chunks} chunks ({progress:.1f}%), entries so far: {total_collected:,}")
            
            # Read chunk
            end_row = min(row_start + row_chunk_size, n_rows)
            indptr_chunk = x_group['indptr'][row_start:end_row + 1]
            
            start_idx = indptr_chunk[0]
            end_idx = indptr_chunk[-1]
            
            if start_idx == end_idx:  # No data in this chunk
                continue
            
            data_chunk = x_group['data'][start_idx:end_idx]
            indices_chunk = x_group['indices'][start_idx:end_idx]
            indptr_chunk = indptr_chunk - start_idx
            
            # Pre-allocate for this chunk
            chunk_rows = np.zeros(max_entries_per_chunk, dtype=np.int32)
            chunk_cols = np.zeros(max_entries_per_chunk, dtype=np.int32)
            chunk_values = np.zeros(max_entries_per_chunk, dtype=np.float64)
            
            # Collect entries from this chunk
            actual_count = count_and_collect_target_entries(
                data_chunk, indices_chunk, indptr_chunk,
                col_start, col_end, row_start,
                chunk_rows, chunk_cols, chunk_values
            )
            
            # Store results (trim to actual size)
            if actual_count > 0:
                all_rows_list.append(chunk_rows[:actual_count])
                all_cols_list.append(chunk_cols[:actual_count])
                all_values_list.append(chunk_values[:actual_count])
    
    # Concatenate all chunks
    if all_rows_list:
        all_rows = np.concatenate(all_rows_list)
        all_cols = np.concatenate(all_cols_list) 
        all_values = np.concatenate(all_values_list)
        print(f"  Collected {len(all_values):,} total entries")
        return all_rows, all_cols, all_values
    else:
        return np.array([]), np.array([]), np.array([])

def sort_by_column_then_row(cols, rows, values):
    """Sort entries by column, then by row within each column."""
    # Use numpy lexsort (much faster than numba bubble sort)
    # lexsort sorts by last key first, so we want (rows, cols)
    sort_indices = np.lexsort((rows, cols))
    
    return cols[sort_indices], rows[sort_indices], values[sort_indices]

def write_columns_to_csc_batch(all_rows, all_cols, all_values, col_start, col_end, 
                              csc_data_offset, zarr_group):
    """Write collected column data to CSC zarr arrays."""
    if len(all_values) == 0:
        # Update indptr for empty columns
        for col_idx in range(col_start, col_end):
            zarr_group['indptr'][col_idx + 1] = csc_data_offset
        return csc_data_offset
    
    print(f"  Sorting {len(all_values):,} entries...")
    
    # Sort by column, then by row
    sorted_cols, sorted_rows, sorted_values = sort_by_column_then_row(all_cols, all_rows, all_values)
    
    print(f"  Writing to zarr...")
    
    # Write to zarr arrays
    end_offset = csc_data_offset + len(sorted_values)
    zarr_group['data'][csc_data_offset:end_offset] = sorted_values
    zarr_group['indices'][csc_data_offset:end_offset] = sorted_rows
    
    # Update indptr - scan through sorted data to find column boundaries
    current_offset = csc_data_offset
    current_col = col_start
    
    i = 0
    while i < len(sorted_cols) and current_col < col_end:
        # Count entries for current_col
        col_entries = 0
        while i < len(sorted_cols) and sorted_cols[i] == current_col:
            col_entries += 1
            i += 1
        
        current_offset += col_entries
        zarr_group['indptr'][current_col + 1] = current_offset
        current_col += 1
    
    # Fill remaining columns with same offset (empty columns)
    while current_col < col_end:
        zarr_group['indptr'][current_col + 1] = current_offset
        current_col += 1
    
    return end_offset

def convert_h5ad_to_zarr(h5ad_path, cols_per_pass=1000, row_chunk_size=50000):
    """Convert h5ad CSR matrix to zarr CSC format using optimized processing."""
    # Check input format and get dimensions
    shape, nnz, data_dtype = check_h5ad_format(h5ad_path)
    n_rows, n_cols = shape
    
    # Create output zarr file path
    zarr_path = Path(h5ad_path).with_suffix('.zarr')
    if zarr_path.exists():
        print(f"Removing existing zarr file: {zarr_path}")
        import shutil
        shutil.rmtree(zarr_path)
    
    print(f"Creating zarr file: {zarr_path}")
    
    # Create zarr group
    root = zarr.open_group(str(zarr_path), mode='w')
    x_group = root.create_group('X')
    
    # Pre-allocate CSC arrays
    zarr_data = x_group.create_dataset('data', shape=(nnz,), dtype=data_dtype, chunks=True)
    zarr_indices = x_group.create_dataset('indices', shape=(nnz,), dtype='int32', chunks=True)  
    zarr_indptr = x_group.create_dataset('indptr', shape=(n_cols + 1,), dtype='int64', chunks=True)
    
    # Initialize indptr
    zarr_indptr[0] = 0
    
    # Store shape and format metadata
    x_group.attrs['shape'] = shape
    x_group.attrs['format'] = 'csc'
    
    # Calculate number of passes
    num_passes = (n_cols + cols_per_pass - 1) // cols_per_pass
    print(f"Processing {n_cols:,} columns in {num_passes} passes ({cols_per_pass:,} columns per pass)")
    print(f"Using {row_chunk_size:,} rows per chunk")
    
    csc_data_offset = 0
    
    # Process columns in batches
    for pass_idx in range(num_passes):
        col_start = pass_idx * cols_per_pass
        col_end = min(col_start + cols_per_pass, n_cols)
        
        print(f"Pass {pass_idx + 1}/{num_passes}: processing columns {col_start:,} to {col_end - 1:,}")
        
        # Collect columns using optimized approach
        all_rows, all_cols, all_values = collect_columns_optimized(
            h5ad_path, col_start, col_end, n_rows, row_chunk_size
        )
        
        # Write to CSC zarr arrays
        csc_data_offset = write_columns_to_csc_batch(
            all_rows, all_cols, all_values, col_start, col_end,
            csc_data_offset, x_group
        )
        
        print(f"  Total entries written so far: {csc_data_offset:,}")
    
    print(f"✓ Successfully created {zarr_path}")
    print(f"  Shape: {shape}")
    print(f"  Format: CSC") 
    print(f"  Non-zeros: {csc_data_offset:,}")
    print(f"  Data type: {data_dtype}")

def main():
    if len(sys.argv) not in [2, 3, 4]:
        print("Usage: python convert_to_zarr.py <h5ad_file> [cols_per_pass] [row_chunk_size]")
        print("  cols_per_pass: columns to process per pass (default: 1000)")
        print("  row_chunk_size: rows to read at once (default: 50000)")
        sys.exit(1)
    
    h5ad_path = sys.argv[1]
    cols_per_pass = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    row_chunk_size = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
    
    if not os.path.exists(h5ad_path):
        print(f"Error: File {h5ad_path} not found")
        sys.exit(1)
    
    if not h5ad_path.endswith('.h5ad'):
        print(f"Error: File must have .h5ad extension")
        sys.exit(1)
    
    print(f"Parameters: {cols_per_pass:,} columns per pass, {row_chunk_size:,} rows per chunk")
    
    try:
        convert_h5ad_to_zarr(h5ad_path, cols_per_pass, row_chunk_size)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()