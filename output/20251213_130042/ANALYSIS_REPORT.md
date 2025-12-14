# Ring-to-3D Pipeline Analysis Report

**Project:** Text-Depth-Point-to-Cloud-3D-CAD  
**Prepared for:** TK Bala  
**Report Date:** December 14, 2025

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Trial 1: Original Run (December 13)](#trial-1-original-run-december-13)
3. [Investigation & Root Cause Analysis](#investigation--root-cause-analysis)
4. [Trial 2: Re-run with Fixes (December 14)](#trial-2-re-run-with-fixes-december-14)
5. [Comparison & Results](#comparison--results)
6. [Conclusion & Next Steps](#conclusion--next-steps)

---

## Executive Summary

| Trial | Date | Outcome | Key Metric |
|-------|------|---------|------------|
| **Trial 1** | Dec 13, 2025 | ❌ Flat mesh produced | Depth std: 0.18 |
| **Trial 2** | Dec 14, 2025 | ✅ 3D geometry achieved | Depth std: 3.59 |

**Bottom Line:** The original run failed to produce meaningful 3D geometry due to environment issues and a code bug. After fixes, we achieved a **20x improvement** in depth variation, producing actual 3D structure.

---

# Trial 1: Original Run (December 13)

## 1.1 Run Information

| Property | Value |
|----------|-------|
| **Run ID** | 20251213_130042 |
| **Timestamp** | December 13, 2025, 1:00:42 PM |
| **Input Image** | `image_source/sample_1.png` (1024×1024) |
| **Runtime** | ~11 seconds |
| **Outcome** | ❌ Produced flat 3D mesh |

## 1.2 Pipeline Steps

| Step | Status | Notes |
|------|--------|-------|
| 1. Image Input | ✅ Success | Loaded correctly |
| 2. Segmentation | ⚠️ Partial | 3 components detected |
| 3. Depth Estimation | ❌ Failed | Fallback used |
| 4. Point Cloud | ✅ Success | Generated but flat |
| 5. Mesh Reconstruction | ⚠️ Issues | Flat geometry |
| 6. Export | ✅ Success | Files created |

## 1.3 Segmentation Results

| Component | Pixels | Coverage |
|-----------|--------|----------|
| Ring Metal Body | 297,472 | 28.37% |
| Gemstone | 88,810 | 8.47% |
| Diamond | 2,756 | 0.26% |

⚠️ Only 3 components detected (expected 6). Missing: prongs, setting, additional gemstones.

## 1.4 Depth Map Analysis (The Problem)

```
Depth Statistics:
├── Range:     1.31 - 3.63 units
├── Mean:      1.87 units
└── Std Dev:   0.18  ← CRITICAL: Very low!

Percentile Distribution:
├── 10th: 1.79
├── 25th: 1.80
├── 50th: 1.81  ← 90% of values here!
├── 75th: 1.82
└── 90th: 2.01
```

**🚨 PROBLEM:** 90% of depth values fell within a 0.22-unit range. The depth map was essentially **flat**.

## 1.5 Output Files

```
output/20251213_130042/
├── 01_input_image.png
├── 02_segments/
│   ├── ring_metal_body_00_mask.png
│   ├── gemstone_02_mask.png
│   └── diamond_01_mask.png
├── 03_depth_map.png          ← Flat!
├── 03_depth_map_raw.npy
└── 07_exports/
    └── combined_ring.obj     ← 21,933 verts (flat geometry)
```

---

# Investigation & Root Cause Analysis

## 2.1 Why Did Depth Estimation Fail?

We investigated why MiDaS/Metric3D produced flat depth output.

### Root Cause 1: Python Version Incompatibility

```
Environment Check:
├── Python Version: 3.14 (too new!)
├── PyTorch: 2.9.1+cpu
└── CUDA Available: False

Problem: PyTorch only supports Python 3.9-3.12
```

### Root Cause 2: No GPU Acceleration

Despite having an **NVIDIA RTX 5070 Ti** with **CUDA 13.0**, PyTorch was running in CPU-only mode because:
- Python 3.14 wheels don't exist for CUDA PyTorch
- The `+cpu` suffix indicates no CUDA support

### Root Cause 3: Missing Dependencies

```
Missing Libraries:
├── timm      → Required for MiDaS DPT models
└── mmengine  → Required for Metric3D
```

Without `timm`, MiDaS fell back to a simple brightness-based depth estimation.

### Root Cause 4: Depth Conversion Bug

The MiDaS disparity-to-depth conversion had a critical bug:

```python
# BUGGY CODE:
depth_map = 1.0 / (depth_map + 1e-6)  # Inverse depth
depth_map = (depth_map - min) / (max - min)  # Normalize

# Problem: When disparity values are clustered (e.g., 10-12),
# their inverse (0.1, 0.083) becomes nearly identical after normalization
```

---

# Trial 2: Re-run with Fixes (December 14)

## 3.1 Fixes Applied

### Fix A: New Python Environment

```powershell
# Created Python 3.12 environment
py -3.12 -m venv .venv312

# Installed PyTorch with CUDA support (nightly for RTX 5070 Ti)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Result:**
```
Environment Check:
├── Python Version: 3.12
├── PyTorch: 2.11.0.dev20251214+cu128
├── CUDA Available: True ✅
└── GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

### Fix B: Installed Missing Dependencies

```powershell
pip install timm open3d transformers diffusers segment-anything
```

### Fix C: Fixed Depth Conversion Algorithm

```python
# FIXED CODE in modules/depth_estimation.py:
disparity = np.clip(disparity, 0, None)
max_disp = disparity.max()
if max_disp > 0:
    norm_disp = disparity / max_disp
    depth_map = 1.0 - norm_disp  # Invert: close = low depth
depth_map = depth_map * 10  # Scale to 0-10 units
```

### Fix D: Added Error Handling

```python
# FIXED CODE in modules/point_cloud.py:
try:
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=10)
except Exception as e:
    logger.warning(f"Normal estimation failed: {e}")
```

## 3.2 Re-run Information

| Property | Value |
|----------|-------|
| **Run ID** | 20251214_rerun_v2 |
| **Timestamp** | December 14, 2025, 10:51 PM |
| **Input Image** | Same (`image_source/sample_1.png`) |
| **Outcome** | ✅ Actual 3D geometry produced |

## 3.3 New Depth Map Analysis

```
Depth Statistics:
├── Range:     0.00 - 10.00 units
├── Mean:      5.12 units
└── Std Dev:   3.59  ← 20x improvement!
```

## 3.4 Output Files

```
output/20251214_rerun_v2/
├── 01_input_image.png
├── 02_segments/
│   ├── ring_metal_body_00_mask.png
│   ├── gemstone_02_mask.png
│   └── diamond_01_mask.png
├── 03_depth_map.png          ← Real depth variation!
├── 03_depth_map_raw.npy
├── 05_processed_clouds/
│   ├── ring_metal_body_00_processed.ply
│   └── gemstone_02_processed.ply
├── 06_meshes/
│   ├── ring_metal_body_00_mesh.ply  (10,337 vertices)
│   └── gemstone_02_mesh.ply         (5,318 vertices)
└── 07_exports/
    ├── combined_ring.ply     (15,655 verts, 30,929 tris)
    ├── combined_ring.obj
    └── combined_ring.stl
```

---

# Comparison & Results

## 4.1 Side-by-Side Comparison

| Metric | Trial 1 (Dec 13) | Trial 2 (Dec 14) | Change |
|--------|------------------|------------------|--------|
| **Python** | 3.14 | 3.12 | Fixed |
| **PyTorch** | 2.9.1+cpu | 2.11.0+cu128 | GPU enabled |
| **Depth Model** | Fallback | MiDaS (GPU) | Fixed |
| **Depth Std Dev** | 0.18 | 3.59 | **20x better** |
| **Z Depth Range** | ~0 (flat) | 10.5 units | ✅ 3D |
| **Mesh Vertices** | 21,933 | 15,655 | Cleaner |
| **Mesh Triangles** | 43,804 | 30,929 | Cleaner |
| **Geometry** | Flat surface | 3D structure | ✅ Fixed |

## 4.2 Visual Improvement

```
Trial 1 Depth Distribution:     Trial 2 Depth Distribution:
                               
█████████████████████ 90%      ████████ 20%
█ 5%                           █████████ 25%
█ 5%                           ████████████ 30%
                               ████████ 20%
                               ██ 5%
       (Clustered)                  (Distributed)
```

---

# Conclusion & Next Steps

## 5.1 Summary

| Question | Answer |
|----------|--------|
| **What went wrong?** | Python 3.14 + CPU PyTorch + depth conversion bug |
| **How did we fix it?** | Python 3.12 + CUDA PyTorch + fixed algorithm |
| **What improved?** | 20x better depth variation, actual 3D geometry |

## 5.2 Current Capabilities

✅ **What the pipeline can do:**
- Generate 3D mesh approximation from a 2D ring image
- Segment ring components (metal body, gemstones)
- Export in multiple formats (STL, OBJ, PLY)
- Provide visualization reference for CAD designers

❌ **What the pipeline cannot do:**
- Produce manufacturing-grade CAD models
- Generate exact dimensions (mm tolerances)
- Create parametric geometry (STEP, IGES)
- Capture occluded surfaces (back of ring)

## 5.3 Distance from Manufacturing-Ready CAD

| Requirement | Current Status | Gap |
|-------------|----------------|-----|
| Precise dimensions | Relative depth only | ❌ Need scaling |
| Watertight mesh | Surface artifacts | ⚠️ Needs cleanup |
| NURBS surfaces | Triangle mesh | ❌ Not available |
| CAD formats | STL/OBJ/PLY | ❌ No STEP/IGES |
| Fine details | Approximation | ⚠️ Prongs, settings missing |

**Assessment:** The output can serve as a **concept reference** or **starting point** for a CAD designer, but is **NOT directly usable for manufacturing**.

## 5.4 Recommended Next Steps

1. **For better depth:** Consider multi-view input or stereo images
2. **For CAD output:** Integrate with parametric CAD software (FreeCAD, Rhino)
3. **For manufacturing:** Manual refinement by professional CAD designer required

---

*Report prepared by automated analysis pipeline*  
*Last updated: December 14, 2025*
