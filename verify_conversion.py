#!/usr/bin/env python3
"""
Verify CSR to CSC conversion by comparing the original and converted matrices.
"""

import h5py
import numpy as np
import json
import scipy.sparse as sp
import time

def load_csr_matrix(h5ad_file):
    """Load CSR matrix from h5ad file using memory mapping."""
    print("Memory-mapping CSR matrix...")
    f = h5py.File(h5ad_file, 'r')  # Keep file open for memmap
    data = f['X/data']
    indices = f['X/indices'] 
    indptr = f['X/indptr']
    shape = f['X'].attrs['shape']
    
    # Note: h5py datasets already act like memory-mapped arrays
    return sp.csr_matrix((data, indices, indptr), shape=shape), f

def load_csc_matrix(csc_dir):
    """Load CSC matrix from binary files using memory mapping."""
    print("Memory-mapping CSC matrix...")
    
    # Load metadata
    with open(f'{csc_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    shape = tuple(metadata['shape'])
    
    # Memory-map arrays
    data = np.memmap(f'{csc_dir}/data.bin', dtype=np.float64, mode='r')
    indices = np.memmap(f'{csc_dir}/indices.bin', dtype=np.int64, mode='r')
    indptr = np.fromfile(f'{csc_dir}/indptr.bin', dtype=np.int64)  # indptr is small, load normally
    
    return sp.csc_matrix((data, indices, indptr), shape=shape)

def compare_matrices(csr_matrix, csc_matrix):
    """Compare CSR and CSC matrices for equality."""
    print("Comparing matrices...")
    
    print(f"CSR shape: {csr_matrix.shape}")
    print(f"CSC shape: {csc_matrix.shape}")
    print(f"CSR nnz: {csr_matrix.nnz:,}")
    print(f"CSC nnz: {csc_matrix.nnz:,}")
    
    if csr_matrix.shape != csc_matrix.shape:
        print("❌ Shapes don't match!")
        return False
    
    if csr_matrix.nnz != csc_matrix.nnz:
        print("❌ Number of non-zeros don't match!")
        return False
    
    # Convert both to dense for comparison (only works for small matrices)
    if csr_matrix.shape[0] * csr_matrix.shape[1] < 1e8:  # Less than 100M elements
        print("Converting to dense for element-wise comparison...")
        csr_dense = csr_matrix.toarray()
        csc_dense = csc_matrix.toarray()
        
        if np.allclose(csr_dense, csc_dense):
            print("✅ Matrices are identical (element-wise)")
            return True
        else:
            print("❌ Matrices differ!")
            # Find first difference
            diff_mask = ~np.isclose(csr_dense, csc_dense)
            if np.any(diff_mask):
                i, j = np.where(diff_mask)
                print(f"First difference at ({i[0]}, {j[0]}): CSR={csr_dense[i[0], j[0]]}, CSC={csc_dense[i[0], j[0]]}")
            return False
    else:
        # For large matrices, just do spot checks
        print("Matrix too large for full comparison, doing spot checks...")
        
        # Check a few random elements
        np.random.seed(42)
        n_checks = 1000
        rows = np.random.randint(0, csr_matrix.shape[0], n_checks)
        cols = np.random.randint(0, csr_matrix.shape[1], n_checks)
        
        for i, (r, c) in enumerate(zip(rows, cols)):
            csr_val = csr_matrix[r, c]
            csc_val = csc_matrix[r, c]
            if not np.isclose(csr_val, csc_val):
                print(f"❌ Difference at ({r}, {c}): CSR={csr_val}, CSC={csc_val}")
                return False
        
        print(f"✅ {n_checks} random elements match")
        return True

def performance_test(csr_matrix, csc_matrix):
    """Test performance of row vs column access."""
    print("\nPerformance test:")
    
    # Test row access (should be faster for CSR)
    print("Testing row access...")
    start_time = time.time()
    for i in range(min(100, csr_matrix.shape[0])):
        _ = csr_matrix[i, :].toarray()
    csr_row_time = time.time() - start_time
    
    start_time = time.time()
    for i in range(min(100, csc_matrix.shape[0])):
        _ = csc_matrix[i, :].toarray()
    csc_row_time = time.time() - start_time
    
    print(f"CSR row access: {csr_row_time:.3f}s")
    print(f"CSC row access: {csc_row_time:.3f}s")
    
    # Test column access (should be faster for CSC)
    print("Testing column access...")
    start_time = time.time()
    for j in range(min(100, csr_matrix.shape[1])):
        _ = csr_matrix[:, j].toarray()
    csr_col_time = time.time() - start_time
    
    start_time = time.time()
    for j in range(min(100, csc_matrix.shape[1])):
        _ = csc_matrix[:, j].toarray()
    csc_col_time = time.time() - start_time
    
    print(f"CSR column access: {csr_col_time:.3f}s")
    print(f"CSC column access: {csc_col_time:.3f}s")
    
    print(f"CSC column speedup: {csr_col_time / csc_col_time:.1f}x")

def main():
    h5ad_file = "test_matrix.h5ad"
    csc_dir = "test_output"
    
    print("=== CSR to CSC Conversion Verification (Memory-Mapped) ===\n")
    
    # Load matrices
    csr_matrix, h5_file = load_csr_matrix(h5ad_file)
    csc_matrix = load_csc_matrix(csc_dir)
    
    try:
        # Compare
        if compare_matrices(csr_matrix, csc_matrix):
            print("\n✅ Conversion verified successfully!")
            
            # Performance test
            performance_test(csr_matrix, csc_matrix)
            
        else:
            print("\n❌ Conversion failed verification!")
    
    finally:
        # Close HDF5 file
        h5_file.close()

if __name__ == "__main__":
    main()