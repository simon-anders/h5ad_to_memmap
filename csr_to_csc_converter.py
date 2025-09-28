#!/usr/bin/env python3
"""
CSR to CSC conversion script using two-pass algorithm with column batching.
Converts h5ad file's X group from CSR to CSC format as binary files in a directory.
"""

import h5py
import numpy as np
import sys
import time
import os
import json
from typing import Tuple
import numba

DATA_DTYPE = ('int16', np.int16)

def load_csr_info(input_file: str) -> Tuple[int, int, int]:
    """Load basic info about the CSR matrix."""
    with h5py.File(input_file, 'r') as f:
        shape = f['X'].attrs['shape']
        n_rows, n_cols = shape[0], shape[1]
        nnz = f['X/data'].shape[0]
    return n_rows, n_cols, nnz

def pass1_count_columns(input_file: str, n_cols: int, chunk_size: int = 10000000) -> np.ndarray:
    """
    Pass 1: Count occurrences of each column index to build CSC indptr.
    """
    print(f"Pass 1: Counting column occurrences...")
    
    col_counts = np.zeros(n_cols, dtype=np.int64)
    
    with h5py.File(input_file, 'r') as f:
        indices = f['X/indices']
        total_nnz = indices.shape[0]
        
        processed = 0
        start_time = time.time()
        
        for start_idx in range(0, total_nnz, chunk_size):
            end_idx = min(start_idx + chunk_size, total_nnz)
            chunk_indices = indices[start_idx:end_idx]
            
            # Count occurrences in this chunk
            np.add.at(col_counts, chunk_indices, 1)
            
            processed += len(chunk_indices)
            elapsed = time.time() - start_time
            rate = processed / elapsed / 1e6
            print(f"  Processed {processed:,} / {total_nnz:,} indices ({rate:.1f}M/s)")
    
    # Convert counts to indptr (cumulative sum with leading zero)
    indptr = np.zeros(n_cols + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(col_counts)
    
    print(f"Pass 1 complete. CSC indptr created.")
    return indptr

@numba.jit(nopython=True, nogil=True)
def fill_csc_chunk_numba(chunk_csr_data, chunk_csr_indices, chunk_csr_indptr, 
                         csc_data, csc_indices, write_ptrs, row_start):
    """
    Numba-compiled function to fill CSC arrays for a row chunk.
    """
    n_rows_in_chunk = len(chunk_csr_indptr) - 1
    
    for local_row_idx in range(n_rows_in_chunk):
        global_row_idx = row_start + local_row_idx
        
        # Get row data boundaries
        row_data_start = chunk_csr_indptr[local_row_idx]
        row_data_end = chunk_csr_indptr[local_row_idx + 1]
        
        # Process non-zeros in this row
        for k in range(row_data_start, row_data_end):
            col_j = chunk_csr_indices[k]
            write_pos = write_ptrs[col_j]
            
            # Write to CSC arrays
            csc_indices[write_pos] = global_row_idx
            csc_data[write_pos] = chunk_csr_data[k]
            
            # Increment write pointer
            write_ptrs[col_j] += 1

def write_metadata(output_dir: str, n_rows: int, n_cols: int, nnz: int) -> None:
    """
    Write CSC matrix metadata to output directory.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metadata (convert numpy types to native Python types for JSON)
    metadata = {
        'encoding-type': 'csc_matrix',
        'encoding-version': '0.1.0', 
        'shape': [int(n_rows), int(n_cols)],
        'nnz': int(nnz),
        'dtypes': {
            'data': DATA_DTYPE[0],
            'indices': 'int64',
            'indptr': 'int64'
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def pass2_fill_csc(input_file: str, output_dir: str, csc_indptr: np.ndarray, 
                   n_rows: int, n_cols: int, nnz: int, 
                   row_chunk_size: int = 10000) -> None:
    """
    Pass 2: Fill CSC data and indices arrays using row chunking.
    """
    print(f"Pass 2: Filling CSC arrays with row chunks of {row_chunk_size}...")
    
    # Create binary files using memmap for efficient writing
    csc_data = np.memmap(os.path.join(output_dir, 'data.bin'),
                         dtype=DATA_DTYPE[1], mode='w+', shape=(nnz,))
    csc_indices = np.memmap(os.path.join(output_dir, 'indices.bin'),
                           dtype=np.int64, mode='w+', shape=(nnz,))
    
    # Write indptr directly
    csc_indptr.tofile(os.path.join(output_dir, 'indptr.bin'))
    
    with h5py.File(input_file, 'r') as f_in:
        # Get CSR datasets
        csr_data = f_in['X/data']
        csr_indices = f_in['X/indices']
        csr_indptr = f_in['X/indptr']
        
        # Initialize write pointers (copy of indptr for all columns)
        write_ptrs = csc_indptr.copy()
        
        # Process rows in chunks
        for row_start in range(0, n_rows, row_chunk_size):
            chunk_start_time = time.time()
            row_end = min(row_start + row_chunk_size, n_rows)
            
            # Load CSR data for this row chunk
            csr_indptr_chunk = csr_indptr[row_start:row_end+1]
            
            # Get the data range for this row chunk
            chunk_data_start = csr_indptr_chunk[0]
            chunk_data_end = csr_indptr_chunk[-1]
            
            if chunk_data_start < chunk_data_end:  # Skip if no data in chunk
                # Load chunk's indices and data
                chunk_csr_indices = csr_indices[chunk_data_start:chunk_data_end]
                chunk_csr_data = csr_data[chunk_data_start:chunk_data_end]
                
                # Adjust indptr to be relative to chunk start
                chunk_csr_indptr = csr_indptr_chunk - chunk_data_start
                
                # Call numba function
                fill_csc_chunk_numba(chunk_csr_data, chunk_csr_indices, chunk_csr_indptr,
                                    csc_data, csc_indices, write_ptrs, row_start)
            
            chunk_time = time.time() - chunk_start_time
            nnz_in_chunk = chunk_data_end - chunk_data_start if chunk_data_start < chunk_data_end else 0
            print(f"    Processed {row_end:,} / {n_rows:,} rows ({chunk_time:.1f}s, {nnz_in_chunk:,} nnz)")
    
    # Ensure data is written to disk
    del csc_data, csc_indices
    
    print(f"Pass 2 complete. CSC matrix written to {output_dir}/")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert CSR to CSC format")
    parser.add_argument("input_file", help="Input h5ad file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--skip-pass1", action="store_true", help="Skip Pass 1 and read existing indptr")
    parser.add_argument("--pass1-chunk-size", type=int, default=10000000, 
                       help="Chunk size for Pass 1 column counting (default: 10000000)")
    parser.add_argument("--pass2-chunk-size", type=int, default=10000,
                       help="Row chunk size for Pass 2 processing (default: 10000)")
    
    args = parser.parse_args()
    
    input_file = args.input_file
    output_dir = args.output_dir
    skip_pass1 = args.skip_pass1
    pass1_chunk_size = args.pass1_chunk_size
    pass2_chunk_size = args.pass2_chunk_size
    
    print(f"Converting {input_file} from CSR to CSC format")
    print(f"Output directory: {output_dir}/")
    
    # Load matrix info
    n_rows, n_cols, nnz = load_csr_info(input_file)
    print(f"Matrix: {n_rows:,} rows × {n_cols:,} cols, {nnz:,} non-zeros")
    
    # Write metadata early to catch permission issues
    write_metadata(output_dir, n_rows, n_cols, nnz)
    
    if skip_pass1:
        # Read existing indptr
        print("Skipping Pass 1, reading existing indptr...")
        indptr_file = os.path.join(output_dir, 'indptr.bin')
        if not os.path.exists(indptr_file):
            print(f"Error: {indptr_file} not found. Run without --skip-pass1 first.")
            sys.exit(1)
        csc_indptr = np.fromfile(indptr_file, dtype=np.int64)
        print(f"Loaded indptr with {len(csc_indptr)} elements")
        pass1_time = 0
    else:
        # Pass 1: Build CSC indptr
        start_time = time.time()
        csc_indptr = pass1_count_columns(input_file, n_cols, pass1_chunk_size)
        pass1_time = time.time() - start_time
        print(f"Pass 1 took {pass1_time:.1f} seconds")
    
    # Pass 2: Fill CSC arrays
    start_time = time.time()
    pass2_fill_csc(input_file, output_dir, csc_indptr, n_rows, n_cols, nnz, pass2_chunk_size)
    pass2_time = time.time() - start_time
    print(f"Pass 2 took {pass2_time:.1f} seconds")
    
    total_time = pass1_time + pass2_time
    print(f"Total conversion time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

if __name__ == "__main__":
    main()