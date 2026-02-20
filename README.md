# char2stl

Convert photos of figurines and characters into 3D-printable STL files using [TRELLIS.2](https://github.com/microsoft/TRELLIS).

Takes a single photo as input and produces a watertight, print-ready mesh — scaled, centered, and repaired for FDM/resin printers.

## Pipeline

```
Photo (JPG/PNG) --> TRELLIS.2 inference --> Mesh cleaning & decimation --> Watertight repair --> Scaled STL
```

1. **TRELLIS.2 inference** — Microsoft's image-to-3D model generates a raw mesh from a single photo
2. **Mesh cleaning** — Hole filling, non-manifold repair, duplicate removal, and decimation via cumesh (GPU-accelerated)
3. **Watertight repair** — pymeshlab ensures the mesh is manifold, coherently oriented, and closed
4. **Scaling** — Mesh is scaled to the target height (default 50mm), centered on XY, and placed on the Z=0 build plate

## Requirements

- NVIDIA GPU with CUDA support (tested on Ampere/sm_86)
- ~16GB+ VRAM for the 4B parameter model
- Python 3.10
- Conda environment with TRELLIS.2 and dependencies

### Dependencies

| Package | Purpose |
|---------|---------|
| [TRELLIS.2](https://github.com/microsoft/TRELLIS) | Image-to-3D model (microsoft/TRELLIS.2-4B) |
| [cumesh](https://github.com/microsoft/TRELLIS) | GPU-accelerated mesh operations |
| [trimesh](https://trimesh.org/) | Mesh I/O and manipulation |
| [pymeshlab](https://pymeshlab.readthedocs.io/) | Watertight mesh repair |
| [flash-attn](https://github.com/Dao-AILab/flash-attention) | Efficient attention (required by TRELLIS.2) |
| PyTorch 2.10+ | ML framework with CUDA support |

## Installation

1. Set up the TRELLIS.2 environment following the [TRELLIS installation guide](https://github.com/microsoft/TRELLIS#installation).

2. Install additional dependencies in the same conda environment:

```bash
conda activate trellis2
pip install trimesh pymeshlab
```

3. Clone this repo:

```bash
git clone https://github.com/coreyhahn/char2stl.git
cd char2stl
```

## Usage

### Basic

```bash
python char2stl.py photo.jpg -o figurine.stl
```

### Batch processing

```bash
python char2stl.py img1.jpg img2.png img3.jpg --batch
```

This outputs `img1.stl`, `img2.stl`, `img3.stl` alongside the inputs.

### All options

```
usage: char2stl.py [-h] [-o OUTPUT] [--resolution {512,1024,1536}]
                   [--decimate DECIMATE] [--scale SCALE] [--seed SEED]
                   [--no-repair] [--batch]
                   input [input ...]

Convert figurine photos to 3D-printable STL files using TRELLIS.2.

positional arguments:
  input                 Input image path(s)

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output STL path (single input only)
  --resolution {512,1024,1536}
                        Voxel resolution (default: 1024)
  --decimate DECIMATE   Target face count (default: 1000000)
  --scale SCALE         Target height in mm (default: 50.0)
  --seed SEED           Random seed (default: 42)
  --no-repair           Skip watertight mesh repair
  --batch               Process multiple inputs, output alongside each
```

### Options explained

| Option | Default | Description |
|--------|---------|-------------|
| `--resolution` | 1024 | Voxel grid resolution for TRELLIS.2. Higher = more detail but slower. 512 is fast/rough, 1536 is slow/detailed. |
| `--decimate` | 1000000 | Target face count after decimation. Lower = smaller file, less detail. 500k is good for most prints. |
| `--scale` | 50.0 | Height of the tallest dimension in mm. The mesh is centered on XY with the bottom at Z=0. |
| `--seed` | 42 | Random seed for reproducible generation. Different seeds produce different meshes from the same photo. |
| `--no-repair` | off | Skip the pymeshlab watertight repair pass. Use if pymeshlab is not installed or you want raw output. |

## Tips for best results

- **Use a clean background** — solid white or neutral backgrounds work best. TRELLIS.2 includes the background in the 3D model.
- **Single subject** — one figurine/character per photo.
- **Good lighting** — even lighting without harsh shadows helps the model understand the geometry.
- **Multiple angles** — if one angle doesn't produce good results, try a different photo of the same subject.
- **Seed variation** — try different `--seed` values (e.g., 0, 1, 2, ...) to get different mesh interpretations.

## Viewing STL files

Upload the output `.stl` file to [ViewSTL.com](https://www.viewstl.com/) to preview it in your browser before printing.

## Troubleshooting

### Out of memory during first run

TRELLIS.2 JIT-compiles CUDA kernels on first use. Limit parallel compilation jobs:

```bash
MAX_JOBS=4 python char2stl.py photo.jpg -o output.stl
```

Subsequent runs use cached kernels and won't need this.

### CUDA errors in post-processing

If you see `cudaErrorInvalidConfiguration` or `invalid device ordinal` errors, try:

```bash
CUDA_LAUNCH_BLOCKING=1 python char2stl.py photo.jpg -o output.stl
```

### flash-attn installation fails (OOM)

flash-attn is memory-intensive to compile. Limit parallel jobs:

```bash
MAX_JOBS=1 pip install flash-attn --no-build-isolation
```

### pymeshlab not found

The watertight repair step is optional. Without pymeshlab, the script prints a warning and skips it. The mesh will still work but may have small holes or non-manifold edges. Install it with:

```bash
pip install pymeshlab
```

## License

MIT
