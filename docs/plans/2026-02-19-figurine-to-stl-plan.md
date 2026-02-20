# Figurine Image to 3D-Printable STL — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the existing char2stl silhouette extruder with a TRELLIS.2-powered tool that takes figurine photos and outputs 3D-printable STL files.

**Architecture:** CLI tool wrapping TRELLIS.2 inference. Image loads via PIL, TRELLIS.2 generates a 3D mesh from a single photo, pymeshlab repairs it for 3D printing, trimesh exports to STL. Runs inside the trellis2 conda environment.

**Tech Stack:** TRELLIS.2 (Microsoft, MIT), PyTorch 2.6/CUDA 12.4, trimesh, pymeshlab, conda

**Hardware:** RTX 3090 24GB on localhost (darthplagueis). ~18GB free VRAM after desktop. CUDA driver 12.8.

---

## Task 1: Clone and Set Up TRELLIS.2

**Files:**
- Create: `/home/cah/r2d2/code/TRELLIS.2/` (git clone)

**Step 1: Clone the repo**

```bash
cd /home/cah/r2d2/code
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
```

This will take a while (large repo with submodules).

**Step 2: Run the setup script**

```bash
cd /home/cah/r2d2/code/TRELLIS.2
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
```

This creates the `trellis2` conda environment with Python 3.10, PyTorch 2.6.0+cu124, and compiles custom CUDA extensions. Takes 10-30 minutes.

**Step 3: Install pymeshlab in the trellis2 env**

```bash
conda activate trellis2
pip install pymeshlab
```

**Step 4: Verify the environment**

```bash
conda activate trellis2
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
from trellis2.pipelines import Trellis2ImageTo3DPipeline
print('trellis2 import OK')
import o_voxel
print('o_voxel import OK')
import trimesh
print('trimesh import OK')
import pymeshlab
print('pymeshlab import OK')
"
```

Expected: All imports succeed, CUDA available, RTX 3090 detected.

**Step 5: Download model weights (first-run cache)**

```bash
conda activate trellis2
python -c "
from trellis2.pipelines import Trellis2ImageTo3DPipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
print('Model downloaded OK')
"
```

This downloads ~8GB of weights from HuggingFace. Only needed once.

---

## Task 2: Write the New char2stl.py

**Files:**
- Replace: `/home/cah/r2d2/code/char2stl/char2stl.py`

**Step 1: Write the full script**

Replace `char2stl.py` with the new TRELLIS.2-based pipeline:

```python
#!/usr/bin/env python3
"""
char2stl.py — Convert a figurine photo to a 3D-printable STL file using TRELLIS.2.

Pipeline: image → TRELLIS.2 inference → mesh post-processing → STL

Usage:
    python char2stl.py input.png -o output.stl
    python char2stl.py photo.jpg --resolution 512 --decimate 500000
    python char2stl.py img1.jpg img2.png --batch
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel


def load_pipeline(model_id="microsoft/TRELLIS.2-4B"):
    """Load TRELLIS.2 pipeline onto GPU."""
    print(f"Loading TRELLIS.2 model ({model_id})...")
    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
    pipeline.cuda()
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    return pipeline


def generate_mesh(pipeline, image_path, resolution=1024, seed=42):
    """Run TRELLIS.2 inference on a single image. Returns a trimesh object."""
    print(f"Processing {image_path}...")
    image = Image.open(image_path)
    print(f"  Image: {image.size[0]}x{image.size[1]}")

    t0 = time.time()
    mesh = pipeline.run(image, seed=seed)[0]
    mesh.simplify(16777216)  # nvdiffrast face count limit
    print(f"  Inference: {time.time() - t0:.1f}s")

    return mesh


def mesh_to_trimesh(mesh, decimate=1000000):
    """Convert TRELLIS.2 MeshWithVoxel to a trimesh object via o_voxel."""
    t0 = time.time()
    result = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimate,
        texture_size=1,  # no texture needed for STL
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False
    )
    # to_glb returns a trimesh Scene or Trimesh; force to single mesh
    if isinstance(result, trimesh.Scene):
        result = trimesh.util.concatenate(result.dump())
    print(f"  Post-process: {time.time() - t0:.1f}s, {len(result.faces)} faces")
    return result


def repair_mesh(mesh):
    """Make mesh watertight and 3D-print-ready using pymeshlab."""
    try:
        import pymeshlab
    except ImportError:
        print("  WARNING: pymeshlab not installed, skipping repair")
        return mesh

    t0 = time.time()
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(mesh.vertices, mesh.faces)
    ms.add_mesh(m)

    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=100)
    ms.meshing_re_orient_faces_coherentely()

    repaired = ms.current_mesh()
    result = trimesh.Trimesh(
        vertices=repaired.vertex_matrix(),
        faces=repaired.face_matrix()
    )
    print(f"  Repair: {time.time() - t0:.1f}s, {len(result.faces)} faces")
    return result


def scale_mesh(mesh, target_height_mm=50.0):
    """Scale mesh so its tallest dimension equals target_height_mm."""
    bbox = mesh.bounds
    current_size = bbox[1] - bbox[0]
    max_dim = current_size.max()
    if max_dim == 0:
        return mesh
    scale_factor = target_height_mm / max_dim
    mesh.apply_scale(scale_factor)
    # Center XY, put bottom at Z=0
    bbox = mesh.bounds
    center = (bbox[0] + bbox[1]) / 2
    mesh.apply_translation([-center[0], -center[1], -bbox[0][2]])
    return mesh


def process_one(pipeline, image_path, output_path, args):
    """Full pipeline for one image."""
    mesh = generate_mesh(pipeline, image_path, resolution=args.resolution, seed=args.seed)
    tmesh = mesh_to_trimesh(mesh, decimate=args.decimate)
    if not args.no_repair:
        tmesh = repair_mesh(tmesh)
    tmesh = scale_mesh(tmesh, target_height_mm=args.scale)

    tmesh.export(str(output_path))
    file_size = output_path.stat().st_size
    bbox = tmesh.bounds
    size = bbox[1] - bbox[0]
    print(f"  Wrote {output_path} ({file_size:,} bytes, {len(tmesh.faces)} faces)")
    print(f"  Dimensions: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm")


def main():
    parser = argparse.ArgumentParser(
        description="Convert figurine photos to 3D-printable STL files using TRELLIS.2."
    )
    parser.add_argument("input", nargs="+", help="Input image path(s)")
    parser.add_argument("-o", "--output", help="Output STL path (single input only)")
    parser.add_argument("--resolution", type=int, default=1024, choices=[512, 1024, 1536],
                        help="Voxel resolution (default: 1024)")
    parser.add_argument("--decimate", type=int, default=1000000,
                        help="Target face count (default: 1000000)")
    parser.add_argument("--scale", type=float, default=50.0,
                        help="Target height in mm (default: 50.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-repair", action="store_true",
                        help="Skip watertight mesh repair")
    parser.add_argument("--batch", action="store_true",
                        help="Process multiple inputs, output alongside each")

    args = parser.parse_args()

    if len(args.input) > 1 and args.output:
        print("Error: -o/--output cannot be used with multiple inputs. Use --batch.", file=sys.stderr)
        sys.exit(1)

    if len(args.input) > 1 and not args.batch:
        print("Error: multiple inputs require --batch flag.", file=sys.stderr)
        sys.exit(1)

    # Validate inputs exist
    input_paths = []
    for p in args.input:
        path = Path(p)
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)
        input_paths.append(path)

    # Load model once
    pipeline = load_pipeline()

    # Process each input
    for input_path in input_paths:
        if args.output and len(input_paths) == 1:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_suffix(".stl")

        process_one(pipeline, input_path, output_path, args)

    print("Done.")


if __name__ == "__main__":
    main()
```

**Step 2: Verify syntax**

```bash
conda run -n trellis2 python -m py_compile /home/cah/r2d2/code/char2stl/char2stl.py
```

Expected: No output (clean compile).

**Step 3: Verify help text**

```bash
conda run -n trellis2 python /home/cah/r2d2/code/char2stl/char2stl.py --help
```

Expected: Shows usage with all arguments.

---

## Task 3: Test on the Three Figurine Images

**Files:**
- Read: `/home/cah/r2d2/code/char2stl/403627A3-A3B6-4650-9146-200E1D1BE121.png`
- Read: `/home/cah/r2d2/code/char2stl/403627A3-A3B6-4650-9146-200E1D1BE121.jpg`
- Read: `/home/cah/r2d2/code/char2stl/AC6584CD-FAE5-425A-9EFA-1CD14E8A487F.jpg`

**Step 1: Test on one image (scholar PNG) at low resolution first**

Start with 512 resolution to verify the pipeline works with available VRAM:

```bash
cd /home/cah/r2d2/code/char2stl
conda run -n trellis2 python char2stl.py "403627A3-A3B6-4650-9146-200E1D1BE121.png" \
    -o test_scholar.stl --resolution 512
```

Expected: Produces `test_scholar.stl`, prints timing and mesh stats. If OOM, we know to stick with 512 or close desktop apps.

**Step 2: Test at 1024 resolution if 512 worked**

```bash
conda run -n trellis2 python char2stl.py "403627A3-A3B6-4650-9146-200E1D1BE121.png" \
    -o test_scholar_1024.stl --resolution 1024
```

Expected: Higher quality mesh. May take 30-60s. If OOM, 512 is our max on this GPU.

**Step 3: Batch process all three images**

```bash
conda run -n trellis2 python char2stl.py \
    "403627A3-A3B6-4650-9146-200E1D1BE121.png" \
    "403627A3-A3B6-4650-9146-200E1D1BE121.jpg" \
    "AC6584CD-FAE5-425A-9EFA-1CD14E8A487F.jpg" \
    --batch
```

Expected: Produces three `.stl` files alongside the input images.

**Step 4: Validate STL files**

```bash
conda run -n trellis2 python -c "
import trimesh
for f in ['test_scholar.stl', '403627A3-A3B6-4650-9146-200E1D1BE121.stl',
          'AC6584CD-FAE5-425A-9EFA-1CD14E8A487F.stl']:
    try:
        m = trimesh.load(f)
        print(f'{f}: {len(m.faces)} faces, watertight={m.is_watertight}, '
              f'volume={m.volume:.1f} mm³, bounds={m.bounds[1] - m.bounds[0]}')
    except Exception as e:
        print(f'{f}: ERROR - {e}')
"
```

Expected: All files load, most are watertight, have reasonable volumes and dimensions (~50mm tall).

---

## Task 4: Clean Up Old Files and Commit

**Files:**
- Delete: `/home/cah/r2d2/code/char2stl/test_scholar_png.stl` (old test output)
- Delete: `/home/cah/r2d2/code/char2stl/test_scholar_jpg.stl` (old test output)
- Delete: `/home/cah/r2d2/code/char2stl/test_warrior.stl` (old test output)

**Step 1: Remove old test STL files from v1**

```bash
cd /home/cah/r2d2/code/char2stl
rm -f test_scholar_png.stl test_scholar_jpg.stl test_warrior.stl
```

**Step 2: Initialize git repo (project isn't one yet)**

```bash
cd /home/cah/r2d2/code/char2stl
git init
```

**Step 3: Create .gitignore**

```
__pycache__/
*.stl
*.pyc
```

**Step 4: Commit**

```bash
git add char2stl.py .gitignore docs/
git commit -m "feat: rewrite char2stl to use TRELLIS.2 for figurine-to-STL conversion

Replaces the old silhouette extrusion pipeline with TRELLIS.2-based
single-image 3D reconstruction for generating 3D-printable STL files
from figurine photographs."
```
