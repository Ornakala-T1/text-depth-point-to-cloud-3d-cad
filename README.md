# Ring to 3D Pipeline 💍 → 🔲

A complete Python pipeline that generates manufacturing-ready 3D models of rings from text prompts.

## Overview

This pipeline takes a simple text description of a ring and automatically:
1. **Generates a high-quality image** using Stable Diffusion
2. **Segments ring components** (metal, gemstones, prongs) using Grounded-SAM
3. **Estimates depth** using Metric3D
4. **Creates point clouds** from depth maps
5. **Reconstructs meshes** using Poisson surface reconstruction
6. **Exports to manufacturing formats** (STL, OBJ, PLY, 3DM)

## Quick Start

```bash
# 1. Clone and setup
cd Bala
pip install -r requirements.txt

# 2. Download model checkpoints (see below)

# 3. Run with a prompt
python run.py "elegant diamond engagement ring with gold band"
```

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA (recommended) or CPU
- ~10GB disk space for models

### Step 1: Create Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 2: Install PyTorch with CUDA
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Model Checkpoints

**SAM (Segment Anything):**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**Grounding DINO (optional, for better segmentation):**
```bash
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Usage

### Basic Usage
```bash
python run.py "your ring description here"
```

### With Custom Output Name
```bash
python run.py "platinum wedding band with diamonds" --output my_ring
```

### Python API
```python
from ring_to_3d_pipeline import RingTo3DPipeline
from utils.config import PipelineConfig

# Create pipeline
config = PipelineConfig(device="cuda")
pipeline = RingTo3DPipeline(config=config)

# Generate 3D model
results = pipeline.run(
    prompt="elegant diamond engagement ring with gold band",
    run_name="my_ring"
)

print(f"Output saved to: {results['output_dir']}")
```

## Pipeline Steps

### Step 1: Image Generation
- Uses Stable Diffusion 2.1
- Prompt engineering for consistent camera angles
- Optimized for 3D reconstruction quality

### Step 2: Segmentation (Grounded-SAM)
- Grounding DINO for text-based detection
- SAM for precise mask generation
- Separates: metal body, gemstones, prongs, settings

### Step 3: Depth Estimation (Metric3D)
- Zero-shot metric depth from single image
- Also estimates surface normals
- Supports multiple model sizes (vit_small, vit_large, vit_giant2)

### Step 4: Point Cloud Generation
- Converts depth + camera intrinsics to 3D points
- Formula: `3D_point = depth × inverse(camera_matrix) × pixel_coordinates`
- Separate point clouds per component

### Step 5: Point Cloud Processing
- Outlier removal (statistical + radius)
- Voxel downsampling
- Normal estimation
- Uses Open3D and point-cloud-utils

### Step 6: Mesh Reconstruction
- Poisson surface reconstruction (default)
- Ball Pivoting Algorithm (optional)
- Alpha Shapes (optional)
- Mesh smoothing and cleaning

### Step 7: Export
- **STL**: Universal 3D printing format
- **OBJ**: Common interchange format
- **PLY**: Point cloud and mesh format
- **3DM**: Rhino/Grasshopper format (requires rhino3dm)
- **STEP**: CAD format (requires pythonocc)

## Output Structure

```
output/
└── run_name/
    ├── 01_generated_image.png      # AI-generated ring image
    ├── 02_segments/                 # Segmentation masks
    │   ├── ring_metal_mask.png
    │   ├── gemstone_mask.png
    │   └── segmentation_viz.png
    ├── 03_depth_map.png            # Depth visualization
    ├── 03_depth_map_raw.npy        # Raw depth data
    ├── 04_point_clouds/            # Per-component point clouds
    │   ├── ring_metal_pointcloud.ply
    │   └── gemstone_pointcloud.ply
    ├── 05_processed_clouds/        # Cleaned point clouds
    ├── 06_meshes/                  # Reconstructed meshes
    │   ├── ring_metal_mesh.ply
    │   └── gemstone_mesh.ply
    └── 07_exports/                 # Manufacturing formats
        ├── ring_metal.stl
        ├── ring_metal.obj
        ├── gemstone.stl
        └── combined_ring.stl
```

## Configuration

Edit `utils/config.py` to customize:

```python
@dataclass
class PipelineConfig:
    device: str = "cuda"              # or "cpu"
    image_size: int = 512             # Generated image size
    sd_model_id: str = "stabilityai/stable-diffusion-2-1"
    depth_model: str = "vit_small"    # vit_small, vit_large, vit_giant2
    mesh_algorithm: str = "poisson"   # poisson, ball_pivoting, alpha_shape
```

## Prompt Tips

For best results, describe:
- **Metal type**: gold, white gold, platinum, silver
- **Stone type**: diamond, sapphire, ruby, emerald
- **Style**: solitaire, halo, three-stone, pavé
- **Details**: prongs, setting style, band width

**Good prompts:**
- "elegant solitaire diamond engagement ring with platinum band"
- "vintage rose gold ring with oval sapphire and diamond halo"
- "modern white gold wedding band with channel-set diamonds"

## References

This pipeline uses the following open-source projects:
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [Metric3D](https://github.com/YvanYin/Metric3D)
- [point-cloud-utils](https://github.com/fwilliams/point-cloud-utils)
- [Open3D](http://www.open3d.org/)
- [ControlNet](https://github.com/lllyasviel/ControlNet)

## Troubleshooting

### CUDA out of memory
- Use a smaller depth model: `depth_model="vit_small"`
- Reduce image size: `image_size=384`
- Use CPU (slower): `device="cpu"`

### Model download fails
- Check internet connection
- Models are downloaded automatically on first run
- Or download manually (see Installation)

### Poor segmentation
- Ensure clear background in generated images
- Try different prompts for cleaner images
- Download full Grounded-SAM for better results

### Mesh has holes
- Enable watertight post-processing
- Use Poisson reconstruction (default)
- Increase depth estimation quality

## License

This project is for educational and research purposes. Individual components have their own licenses:
- Stable Diffusion: [License](https://github.com/Stability-AI/stablediffusion/blob/main/LICENSE)
- SAM: Apache 2.0
- Grounding DINO: Apache 2.0
- Metric3D: BSD-2-Clause
