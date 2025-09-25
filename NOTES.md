# H5AD to Zarr Conversion - Project Notes

## Goal
Convert large h5ad file (`Mouse_Aging_10Xv3_counts_20241115.h5ad`) from CSR to CSC format in zarr for efficient gene-wise access.

## Current Status
- **Dataset:** 1,162,565 cells × 32,285 genes, 5.7B non-zeros, float64
- **Current approach:** Multi-column passes with chunked reading
- **Problem:** Very slow - I/O bound, ~25 seconds per chunk

## Files Created
- `convert_to_zarr_v7.py` - Latest version (numba + optimizations)
- Previous versions: v1-v6 (various approaches, mostly failed due to memory issues)

## Algorithm Evolution

### V1-V4: Row-wise processing (FAILED - memory issues)
- Tried to collect all triplets in memory → OOM
- Tried streaming to pickle files → still too much memory
- Problem: Need to scatter data across 32k columns

### V5-V6: Multi-column passes (SLOW - I/O bound)
- Process 1000 columns per pass (33 passes total)
- Each pass scans entire 1.2M row dataset
- V6 added numba optimization
- Problem: 33 × 10+ minutes = 5+ hours total

### V7: Current version optimizations
- Larger chunks (50k rows instead of 10k)
- Persistent file handle (no open/close overhead)  
- Single-pass collection (no separate counting phase)
- Result: Still slow - 25s per chunk, I/O bound

## Performance Analysis
- **V6:** 10s per 10k-row chunk → ~19 min per pass → 10+ hours total
- **V7:** 25s per 50k-row chunk → similar total time
- **Root cause:** Reading through 1.2M rows × 33 passes = massive I/O

## Key Insight from Today
**Chunked reading might be counterproductive!**
- HDF5 likely has internal caching/buffering
- Our chunked approach creates overhead vs fewer large reads
- Need to test: bulk reads vs chunked reads

## Next Steps to Try
1. **Test bulk reading:** Load entire arrays at once vs chunked
   - `all_data = f['X/data'][:]` (~46GB)
   - `all_indices = f['X/indices'][:]` (~23GB) 
   - May not fit in memory, but test if dramatically faster

2. **Test larger chunks:** Try 500k, 1M row chunks to find sweet spot

3. **Alternative algorithms:**
   - True single-pass (collect all columns simultaneously) - needs ~20GB memory
   - Hybrid: fewer passes with more columns each
   - External sorting approaches

## Technical Details
- **Memory per 1000-column pass:** ~3-5GB (manageable)
- **Total dataset size:** ~91GB if fully loaded
- **Numba optimizations:** Already implemented for hot loops
- **Target format:** CSC in zarr with proper indptr/indices/data arrays

## Open Questions
- Is HDF5 chunking layout optimal for our access pattern?
- Would memory-mapping help vs explicit reads?
- Can we process in column-major order somehow?

## Command to Resume
```bash
cd /home/anders/w/h5ad_to_zarr
python convert_to_zarr_v7.py Mouse_Aging_10Xv3_counts_20241115.h5ad
```

## Files in Directory
- `Mouse_Aging_10Xv3_counts_20241115.h5ad` - Source data
- `convert_to_zarr_v7.py` - Current working script  
- `NOTES.md` - This file
- Various older script versions (v1-v6)