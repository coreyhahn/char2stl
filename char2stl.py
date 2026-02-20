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

import torch
import numpy as np
import trimesh
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import cumesh


def load_pipeline(model_id="microsoft/TRELLIS.2-4B"):
    """Load TRELLIS.2 pipeline onto GPU."""
    print(f"Loading TRELLIS.2 model ({model_id})...")
    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
    pipeline.cuda()
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    return pipeline


def generate_mesh(pipeline, image_path, resolution=1024, seed=42):
    """Run TRELLIS.2 inference on a single image. Returns a MeshWithVoxel."""
    print(f"Processing {image_path}...")
    image = Image.open(image_path)
    print(f"  Image: {image.size[0]}x{image.size[1]}")

    t0 = time.time()
    mesh = pipeline.run(image, seed=seed)[0]
    mesh.simplify(16777216)  # nvdiffrast face count limit
    print(f"  Inference: {time.time() - t0:.1f}s")

    return mesh


def mesh_to_trimesh(mesh, decimate=1000000):
    """Convert TRELLIS.2 MeshWithVoxel to a trimesh object (geometry only, no texture)."""
    t0 = time.time()
    vertices = mesh.vertices.cuda()
    faces = mesh.faces.cuda()

    # Use cumesh for cleaning, remeshing, and decimation
    cm = cumesh.CuMesh()
    cm.init(vertices, faces)
    cm.fill_holes(max_hole_perimeter=3e-2)
    cm.remove_duplicate_faces()
    cm.repair_non_manifold_edges()
    cm.remove_small_connected_components(1e-5)
    cm.fill_holes(max_hole_perimeter=3e-2)
    cm.simplify(decimate)
    cm.remove_duplicate_faces()
    cm.repair_non_manifold_edges()
    cm.unify_face_orientations()

    out_verts, out_faces = cm.read()
    result = trimesh.Trimesh(
        vertices=out_verts.cpu().numpy(),
        faces=out_faces.cpu().numpy(),
    )
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
    ms.meshing_remove_duplicate_faces()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_re_orient_faces_coherently()
    ms.meshing_close_holes(maxholesize=100)

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
