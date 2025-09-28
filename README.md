# h5ad_to_memmap

Tools for converting h5ad files (AnnData format) to memory-mapped binary formats for efficient data access and matrix format conversion.

## Overview

This repository provides Python scripts for working with large sparse matrices stored in h5ad files, particularly useful for single-cell RNA sequencing data. The tools focus on efficient memory usage and fast I/O operations through memory mapping.

## Scripts

### `h5ad_to_memmap.py`

Extracts the sparse CSR matrix from an h5ad file and writes it as memory-mapped binary files, preserving the original CSR format.

**Usage:**
```bash
python h5ad_to_memmap.py input.h5ad output_dir/
```

**Features:**
- Preserves original data types from the h5ad file
- Chunked processing for memory efficiency
- Writes three binary files: `data.bin`, `indices.bin`, `indptr.bin`
- Includes metadata in JSON format

### `csr_to_csc_converter.py`

Converts sparse matrices from CSR (Compressed Sparse Row) to CSC (Compressed Sparse Column) format using a memory-efficient two-pass algorithm.

**Usage:**
```bash
python csr_to_csc_converter.py input.h5ad output_dir/
```

**Features:**
- Two-pass algorithm optimized with Numba for performance
- Memory-efficient processing of large matrices
- Configurable chunk sizes for both passes
- Progress reporting during conversion

**Options:**
- `--skip-pass1`: Skip column counting and use existing indptr
- `--pass1-chunk-size`: Chunk size for column counting (default: 10M)
- `--pass2-chunk-size`: Row chunk size for processing (default: 10k)

### `verify_conversion.py`

Verifies the correctness of matrix conversions by comparing original and converted matrices.

**Usage:**
```bash
python verify_conversion.py
```

**Features:**
- Element-wise comparison for small matrices
- Random sampling verification for large matrices
- Performance benchmarking (row vs column access)
- Memory-mapped loading for efficient comparison

### `create_test_h5ad.py`

Generates synthetic h5ad files for testing the conversion tools.

**Usage:**
```bash
python create_test_h5ad.py
```

**Features:**
- Configurable matrix dimensions and sparsity
- Realistic gene expression-like integer values
- Compressed HDF5 storage

## File Formats

### Memory-Mapped Binary Files

The tools output sparse matrices as separate binary files:

- `data.bin`: Non-zero values (preserves original data type)
- `indices.bin`: Column indices (int64)
- `indptr.bin`: Row pointers (int64)
- `metadata.json`: Matrix metadata including shape, nnz, and data types

### Metadata Format

```json
{
  "encoding-type": "csr_matrix" | "csc_matrix",
  "encoding-version": "0.1.0",
  "shape": [rows, cols],
  "nnz": non_zero_count,
  "dtypes": {
    "data": "int16" | "float64",
    "indices": "int64",
    "indptr": "int64"
  }
}
```

## Performance Considerations

- **CSR format**: Efficient for row-wise operations (common in cell-based analysis)
- **CSC format**: Efficient for column-wise operations (common in gene-based analysis)
- **Memory mapping**: Allows processing matrices larger than available RAM
- **Chunked processing**: Prevents memory exhaustion on large datasets

## Dependencies

- numpy
- scipy
- h5py
- numba

## Installation

```bash
pip install numpy scipy h5py numba
```

## Use Cases

- Converting large single-cell datasets for different analysis workflows
- Preprocessing data for tools that require specific matrix formats
- Creating memory-efficient representations of sparse biological data
- Benchmarking I/O performance for different sparse matrix formats