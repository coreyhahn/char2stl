# char2stl v2: Figurine Image to 3D-Printable STL

**Date:** 2026-02-19
**Status:** Approved

## Problem

The original char2stl extracted 2D silhouette contours and extruded them into flat 3D shapes. The user wants to input photographs of 3D figurines and get actual 3D mesh STL files suitable for 3D printing.

## Approach

Use Microsoft TRELLIS.2 (4B parameter model, MIT license) for single-image 3D reconstruction, running locally on a 24GB+ NVIDIA GPU. Post-process with trimesh and pymeshlab for 3D-print-ready output.

## Architecture

```
figurine.jpg -> [background removal] -> [TRELLIS.2 inference] -> [mesh post-processing] -> output.stl
```

### Pipeline Steps

1. **Load image** - PIL Image from disk
2. **TRELLIS.2 inference** - `pipeline.run(image)` with `preprocess_image=True` (auto background removal + crop)
3. **Simplify** - `mesh.simplify(16777216)` (nvdiffrast face count limit)
4. **Export via o_voxel** - `to_glb()` with `decimation_target` and `remesh=True` -> returns trimesh object
5. **Watertight repair** - pymeshlab filters (close holes, fix normals, remove non-manifold edges)
6. **Scale** - normalize to target size in mm
7. **Export** - `trimesh.export("output.stl")`

## CLI Interface

```
python char2stl.py input.jpg -o output.stl
python char2stl.py input.png --resolution 1024 --decimate 500000
python char2stl.py *.jpg --batch
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | required | One or more image paths |
| `-o / --output` | `<input>.stl` | Output STL path |
| `--resolution` | 1024 | Voxel resolution: 512, 1024, 1536 |
| `--decimate` | 1000000 | Target face count |
| `--scale` | auto (50mm tall) | Output scale in mm |
| `--seed` | 42 | Random seed |
| `--no-repair` | false | Skip watertight repair |
| `--batch` | false | Process all inputs |

## Dependencies

### TRELLIS.2 (external)

Cloned as a sibling directory, provides the conda environment:

```
r2d2/code/
  TRELLIS.2/          # cloned repo with conda env "trellis2"
  char2stl/
    char2stl.py       # our tool (runs inside trellis2 conda env)
```

**Setup:** `git clone --recursive https://github.com/microsoft/TRELLIS.2.git` then run `setup.sh`

### Python packages (within trellis2 conda env)

- trellis2 (from repo) - inference
- trimesh (included) - mesh I/O and STL export
- pymeshlab (to install) - watertight repair, decimation
- PIL/Pillow (included) - image loading

### Hardware

- Linux only
- Python 3.10 (set by TRELLIS.2 setup)
- PyTorch 2.6.0, CUDA 12.4
- 24GB+ VRAM GPU

## TRELLIS.2 API Reference

### Model loading
```python
from trellis2.pipelines import Trellis2ImageTo3DPipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()
```

### Inference
```python
mesh = pipeline.run(image, seed=42)[0]
mesh.simplify(16777216)
```

### Mesh export (to trimesh object)
```python
import o_voxel
glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices,
    faces=mesh.faces,
    attr_volume=mesh.attrs,
    coords=mesh.coords,
    attr_layout=mesh.layout,
    voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=1000000,
    texture_size=4096,
    remesh=True,
    remesh_band=1,
    remesh_project=0,
    verbose=True
)
glb.export("output.stl")
```

### Environment variables (required)
```python
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

## Alternatives Considered

1. **Hunyuan3D 2.1** - Nearly as good quality, lighter VRAM (6GB shape-only), but Tencent community license and less clean topology.
2. **Tripo AI Cloud API** - Simplest implementation, $0.20-0.40/model, but requires internet and ongoing cost.

TRELLIS.2 was chosen for best mesh quality, MIT license, and full local inference on available hardware.
