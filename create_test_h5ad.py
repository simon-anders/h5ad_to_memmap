#!/usr/bin/env python3
"""
Create a test h5ad file with specified dimensions and sparsity for testing CSR to CSC conversion.
"""

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.sparse import random
import time

def create_test_h5ad(filename: str, n_rows: int = 10_000, n_cols: int = 5_000, density: float = 0.05):
    """
    Create a test h5ad file with CSR sparse matrix.
    
    Parameters:
    - filename: Output filename
    - n_rows: Number of rows (cells)
    - n_cols: Number of columns (genes) 
    - density: Fraction of non-zero values (0.05 = 5%)
    """
    
    print(f"Creating test h5ad file: {filename}")
    print(f"Shape: {n_rows:,} × {n_cols:,}")
    print(f"Density: {density*100:.1f}% ({int(n_rows * n_cols * density):,} non-zeros)")
    
    # Generate random sparse matrix in CSR format
    print("Generating random sparse matrix...")
    start_time = time.time()
    
    # Use scipy's random sparse matrix generator
    # Generate with uint16 values (0-65535)
    matrix = random(n_rows, n_cols, density=density, format='csr', dtype=np.float64, random_state=42)
    
    # Convert to uint16 integer values (1-1000 for realistic gene expression)
    matrix.data = np.random.RandomState(42).randint(1, 1001, size=len(matrix.data)).astype(np.uint16)
    
    # Convert back to float64 for h5ad compatibility (anndata typically uses float)
    matrix = matrix.astype(np.float64)
    
    gen_time = time.time() - start_time
    print(f"Matrix generation took {gen_time:.1f} seconds")
    print(f"Actual nnz: {matrix.nnz:,} ({matrix.nnz / (n_rows * n_cols) * 100:.2f}%)")
    
    # Create h5ad file
    print("Writing to HDF5...")
    start_time = time.time()
    
    with h5py.File(filename, 'w') as f:
        # Create X group with CSR format
        x_group = f.create_group('X')
        x_group.attrs['encoding-type'] = 'csr_matrix'
        x_group.attrs['encoding-version'] = '0.1.0'
        x_group.attrs['shape'] = [n_rows, n_cols]
        
        # Write CSR arrays
        x_group.create_dataset('data', data=matrix.data, compression='gzip')
        x_group.create_dataset('indices', data=matrix.indices, compression='gzip') 
        x_group.create_dataset('indptr', data=matrix.indptr, compression='gzip')
        
        # Add some dummy metadata to make it look like a real h5ad
        f.attrs['encoding-type'] = 'anndata'
        f.attrs['encoding-version'] = '0.1.0'
        
        # Create obs (cell metadata) - just dummy names
        obs_group = f.create_group('obs')
        obs_group.attrs['encoding-type'] = 'dataframe'
        obs_group.attrs['encoding-version'] = '0.2.0'
        
        # Create var (gene metadata) - just dummy names  
        var_group = f.create_group('var')
        var_group.attrs['encoding-type'] = 'dataframe'
        var_group.attrs['encoding-version'] = '0.2.0'
    
    write_time = time.time() - start_time
    print(f"HDF5 write took {write_time:.1f} seconds")
    
    # Print file size
    import os
    file_size = os.path.getsize(filename) / 1024**2
    print(f"File size: {file_size:.1f} MB")
    
    print(f"Test file created: {filename}")

if __name__ == "__main__":
    create_test_h5ad("test_matrix.h5ad")