#!/usr/bin/env python3
"""
Extract CSR matrix from h5ad file and write as memory-mapped binary files.
Preserves the original CSR format while converting from HDF5 to binary files.
"""

import h5py
import numpy as np
import sys
import time
import os
import json
from typing import Tuple

def load_csr_info(input_file: str) -> Tuple[int, int, int]:
    """Load basic info about the CSR matrix."""
    with h5py.File(input_file, 'r') as f:
        shape = f['X'].attrs['shape']
        n_rows, n_cols = shape[0], shape[1]
        nnz = f['X/data'].shape[0]
    return n_rows, n_cols, nnz

def write_metadata(output_dir: str, n_rows: int, n_cols: int, nnz: int, data_dtype: str) -> None:
    """Write CSR matrix metadata to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        'encoding-type': 'csr_matrix',
        'encoding-version': '0.1.0',
        'shape': [int(n_rows), int(n_cols)],
        'nnz': int(nnz),
        'dtypes': {
            'data': data_dtype,
            'indices': 'int64',
            'indptr': 'int64'
        }
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def extract_csr_to_memmap(input_file: str, output_dir: str, target_dtype: str, chunk_size: int = 10000000) -> None:
    """
    Extract CSR matrix from h5ad file and write as memory-mapped binary files.
    """
    print(f"Extracting CSR matrix from {input_file}")
    print(f"Output directory: {output_dir}/")

    # Load matrix info
    n_rows, n_cols, nnz = load_csr_info(input_file)
    print(f"Matrix: {n_rows:,} rows × {n_cols:,} cols, {nnz:,} non-zeros")

    with h5py.File(input_file, 'r') as f:
        # Get original data type for comparison
        original_dtype = f['X/data'].dtype
        target_dtype_obj = np.dtype(target_dtype)

        print(f"Original data type: {original_dtype}")
        print(f"Target data type: {target_dtype}")

        # Write metadata
        write_metadata(output_dir, n_rows, n_cols, nnz, target_dtype)

        # Copy indptr (small array, copy directly)
        print("Writing indptr...")
        indptr = f['X/indptr'][:]
        indptr.tofile(os.path.join(output_dir, 'indptr.bin'))

        # Copy indices in chunks
        print("Writing indices...")
        indices_file = os.path.join(output_dir, 'indices.bin')
        indices_memmap = np.memmap(indices_file, dtype=np.int64, mode='w+', shape=(nnz,))

        indices_dataset = f['X/indices']
        for start_idx in range(0, nnz, chunk_size):
            end_idx = min(start_idx + chunk_size, nnz)
            chunk = indices_dataset[start_idx:end_idx]
            indices_memmap[start_idx:end_idx] = chunk
            print(f"  Processed {end_idx:,} / {nnz:,} indices")

        del indices_memmap

        # Copy data in chunks, converting to target dtype
        print("Writing data...")
        data_file = os.path.join(output_dir, 'data.bin')
        data_memmap = np.memmap(data_file, dtype=target_dtype_obj, mode='w+', shape=(nnz,))

        data_dataset = f['X/data']
        for start_idx in range(0, nnz, chunk_size):
            end_idx = min(start_idx + chunk_size, nnz)
            chunk = data_dataset[start_idx:end_idx]
            # Convert chunk to target dtype
            converted_chunk = chunk.astype(target_dtype_obj)
            data_memmap[start_idx:end_idx] = converted_chunk
            print(f"  Processed {end_idx:,} / {nnz:,} data values")

        del data_memmap

    print(f"CSR matrix extracted to {output_dir}/")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract CSR matrix from h5ad to memory-mapped files")
    parser.add_argument("input_file", help="Input h5ad file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--dtype", type=str, required=True,
                       help="Target data type (e.g., float32, float64, int32)")
    parser.add_argument("--chunk-size", type=int, default=10000000,
                       help="Chunk size for copying data (default: 10000000)")

    args = parser.parse_args()

    start_time = time.time()
    extract_csr_to_memmap(args.input_file, args.output_dir, args.dtype, args.chunk_size)
    total_time = time.time() - start_time
    print(f"Total extraction time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

if __name__ == "__main__":
    main()