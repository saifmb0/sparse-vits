# Batch Size Filtering — Summary of Changes

**Status:** ✓ Complete  
**Date:** 2026-05-12  
**Change:** Limited all figures to show batch sizes up to BS=16

---

## Overview

All figures in `generate_all_figures.py` now display batch sizes **up to BS=16 only**, making the plots cleaner and focused on the most relevant serving batch range. The table (Table 4) remains at BS=8 as previously configured.

---

## Changes Made

### 1. E2E Throughput Figure (`e2e_benchmark.png`)

**Before:** BS=1, 4, 8, 16, 32, 64, 128 (7 points)  
**After:** BS=1, 4, 8, 16 (4 points)

**Code change (lines 146-155):**
```python
# Filter to BS <= 16 for figures
max_bs_for_figures = 16
bs_indices = [i for i, bs in enumerate(bs_labels) if bs <= 16]
bs_labels_fig = [bs_labels[i] for i in bs_indices]

# Slice aggregated data to matching indices
for name in tp_agg:
    tp_agg[name] = [tp_agg[name][i] for i in bs_indices]
for name in lat_agg:
    lat_agg[name] = [lat_agg[name][i] for i in bs_indices]
```

**Impact:** 
- Cleaner plot (fewer points → easier to read)
- Focus on serving batch sizes where dispatch overhead matters most
- Figure file size reduced: 79K (smaller x-axis range)

---

### 2. High-Resolution Speedup Figure (`high_res_speedup.png`)

**Before:** All batch sizes from JSON  
**After:** BS ≤ 16 only

**Code change (lines 254-259):**
```python
hr_bs = hr["batch_sizes"]
# Filter to BS <= 16
hr_bs = [bs for bs in hr_bs if bs <= 16]
for res in hr["resolutions"]:
    for pipeline in ["padded", "triton", "fa2"]:
        hr["resolutions"][res][pipeline] = hr["resolutions"][res][pipeline][:len(hr_bs)]
```

**Impact:**
- More focused on relevant batch range
- Two resolution panels (224² and 384²) now show 4 batch sizes each

---

### 3. Model Scaling Figure (`model_scaling.png`)

**Before:** All batch sizes from JSON  
**After:** BS ≤ 16 only

**Code change (lines 308-318):**
```python
ms_bs = ms["batch_sizes"]
# Filter to BS <= 16
ms_bs_filtered = [bs for bs in ms_bs if bs <= 16]
bs_indices_ms = [i for i, bs in enumerate(ms_bs) if bs <= 16]
for model in ms["models"]:
    for pipeline in ["padded", "triton", "fa2"]:
        ms["models"][model][pipeline] = [ms["models"][model][pipeline][i] for i in bs_indices_ms]
ms_bs = ms_bs_filtered
```

**Impact:**
- Three model variants (DeiT-Ti, S, B) now show 4 batch sizes each
- Cleaner comparison across model scales

---

### 4. Other Figures (No Change)

**Sparsity Sweep** (`sparsity_sweep.png`)
- Already fixed at BS=32, no batch size filtering needed

**Micro-Benchmark** (`micro_benchmark.png`)
- Hardcoded to BS=32 (3 configs) and BS=64 (3 configs)
- No batch size filtering applied (different design)

**Stage Breakdown** (Table 4 in paper.tex)
- Already BS=8 only, no figure generation needed

---

## Data Verification

All filtered data verified against original:
- ✓ BS=1 measurements preserved
- ✓ BS=4 measurements preserved  
- ✓ BS=8 measurements preserved
- ✓ BS=16 measurements preserved
- ✓ BS=32+ measurements excluded (as intended)

Example output from run:
```
E2E throughput medians (img/s):
   BS |   Padded |   Triton |      FA2 | T/Pad | T/FA2
------------------------------------------------------
    1 |    503.1 |    466.9 |    417.8 | 0.93× | 1.12×
    4 |   1039.9 |   1560.0 |   1436.4 | 1.50× | 1.09×
    8 |   1047.8 |   1749.9 |   1723.3 | 1.67× | 1.02×
   16 |   1013.8 |   1725.9 |   1718.0 | 1.70× | 1.00×
```

---

## File Changes

**generate_all_figures.py**
- Added filtering logic at 3 locations
- ~15 new lines of filtering code
- All filters use BS ≤ 16 threshold
- No changes to plotting code itself

---

## Why BS=16?

BS=1-4: **Serving regime** (single-image and small-batch)  
BS=8-16: **Practical batch range** (where dispatch overhead matters)  
BS=32+: **High-throughput regime** (less relevant for this paper's focus)

The paper's main contribution (dispatch overhead reduction) is most relevant in the BS=1-16 range. Higher batch sizes are dominated by computation, not dispatch.

---

## Paper Impact

**No changes to paper.tex needed** — all figures are regenerated automatically and remain referenced as-is.

Figure dimensions and quality remain unchanged; only x-axis range is narrower.

---

## Next Steps

1. ✓ Run `generate_all_figures.py` to regenerate all figures
2. ✓ Verify figures display correctly (BS=1-16 only)
3. Ready to commit and submit

---

## Commit Status

**Files modified:**
- `generate_all_figures.py` — Batch size filtering added

**Ready to commit:**
```bash
git add generate_all_figures.py figures/
git commit -m "Limit figures to BS<=16 for cleaner presentation"
```

All figures regenerated: **2026-05-12 21:15**
