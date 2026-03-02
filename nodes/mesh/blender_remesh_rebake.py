"""
Blender remesh + rebake node.

Wraps the CLI script `scripts/blender_remesh_rebake_cli.py` and exposes
quality knobs for triangle targeting, UV strategy, normal smoothing,
floater cleanup, and PBR rebaking.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid

import numpy as np
import torch
import trimesh
from PIL import Image

import folder_paths

from ..texture.common import check_blender


class BlenderRemeshRebake:
    UV_METHODS = ["smart", "angle"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Input GLB/GLTF path. Absolute path or relative to ComfyUI input/output/temp directories."
                }),
                "target_triangles": ("INT", {
                    "default": 250000,
                    "min": 1000,
                    "max": 10000000,
                    "step": 1000,
                    "tooltip": "Target triangle budget for voxel remesh search."
                }),
            },
            "optional": {
                "output_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Output GLB path. If empty, auto-generates in ComfyUI output directory."
                }),
                "tri_tolerance": ("FLOAT", {
                    "default": 0.03,
                    "min": 0.0,
                    "max": 0.9,
                    "step": 0.01,
                    "tooltip": "Accepted relative triangle deviation from target (0.03 = ±3%)."
                }),
                "max_search_steps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Voxel search iterations for matching target triangles."
                }),
                "bake_resolution": ("INT", {
                    "default": 2048,
                    "min": 256,
                    "max": 8192,
                    "step": 256,
                    "tooltip": "Resolution of baked texture maps."
                }),
                "bake_margin": ("INT", {
                    "default": 40,
                    "min": 0,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Bake dilation margin in pixels to reduce seams."
                }),
                "samples": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 1024,
                    "step": 1,
                    "tooltip": "Cycles samples for baking quality."
                }),
                "smooth_angle": ("FLOAT", {
                    "default": 75.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Smoothing angle for remeshed normals."
                }),
                "weighted_normals": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply weighted normals on the remeshed lowpoly."
                }),
                "transfer_source_normals": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Transfer custom split normals from source to output after baking."
                }),
                "remove_floaters": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove small disconnected components after remesh."
                }),
                "floater_min_triangles": ("INT", {
                    "default": 384,
                    "min": 0,
                    "max": 1000000,
                    "step": 1,
                    "tooltip": "Minimum connected-component triangle count to keep."
                }),
                "floater_min_ratio": ("FLOAT", {
                    "default": 0.002,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.0001,
                    "tooltip": "Minimum connected-component ratio of lowpoly triangles to keep."
                }),
                "uv_method": (cls.UV_METHODS, {
                    "default": "smart",
                    "tooltip": "UV unwrap strategy for lowpoly mesh."
                }),
                "uv_angle": ("FLOAT", {
                    "default": 72.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Angle limit for smart UV method."
                }),
                "uv_seam_angle": ("FLOAT", {
                    "default": 65.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Sharp edge threshold used when uv_method=angle."
                }),
                "uv_island_margin": ("FLOAT", {
                    "default": 0.003,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.0005,
                    "tooltip": "UV island margin."
                }),
                "auto_projection": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto compute bake ray/cage projection distances from mesh scale + voxel size."
                }),
                "ray_distance": ("FLOAT", {
                    "default": 0.005,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.0005,
                    "tooltip": "Bake ray distance used when auto_projection=false."
                }),
                "cage_extrusion": ("FLOAT", {
                    "default": 0.0025,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.0005,
                    "tooltip": "Bake cage extrusion used when auto_projection=false."
                }),
                "bake_normal": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Bake normal map."
                }),
                "bake_roughness": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Bake roughness map."
                }),
                "bake_metallic": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Bake metallic map."
                }),
                "bake_ao": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Bake ambient occlusion map."
                }),
                "bake_emission": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Bake emission map."
                }),
                "force_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force Cycles CPU instead of GPU."
                }),
                "timeout_sec": ("INT", {
                    "default": 7200,
                    "min": 60,
                    "max": 43200,
                    "step": 60,
                    "tooltip": "Subprocess timeout for Blender run."
                }),
                "overwrite": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Overwrite output_path if it exists."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("trimesh", "output_path", "albedo", "normal", "roughness", "metallic", "ao")
    OUTPUT_TOOLTIPS = (
        "Remeshed output mesh as TRIMESH.",
        "Output GLB path.",
        "Baked albedo texture.",
        "Baked normal texture.",
        "Baked roughness texture.",
        "Baked metallic texture.",
        "Baked ambient occlusion texture.",
    )
    FUNCTION = "remesh_rebake"
    CATEGORY = "Alvatar/3D"
    DESCRIPTION = (
        "Targeted voxel remesh + PBR rebake using Blender. "
        "Designed for reducing high-poly assets while preserving texture quality."
    )

    def _resolve_existing_path(self, path: str) -> str:
        if not path:
            raise ValueError("file_path is required")

        if os.path.isabs(path) and os.path.exists(path):
            return path

        for base in [
            folder_paths.get_input_directory(),
            folder_paths.get_output_directory(),
            folder_paths.get_temp_directory(),
        ]:
            p = os.path.join(base, path)
            if os.path.exists(p):
                return p

        raise ValueError(f"Input file not found: {path}")

    def _resolve_output_path(self, input_path: str, output_path: str, overwrite: bool) -> str:
        if output_path:
            resolved = output_path if os.path.isabs(output_path) else os.path.join(folder_paths.get_output_directory(), output_path)
        else:
            stem = os.path.splitext(os.path.basename(input_path))[0]
            resolved = os.path.join(folder_paths.get_output_directory(), f"{stem}_remesh_rebake.glb")

        if os.path.splitext(resolved)[1].lower() not in [".glb", ".gltf"]:
            resolved = f"{resolved}.glb"

        os.makedirs(os.path.dirname(resolved), exist_ok=True)

        if (not overwrite) and os.path.exists(resolved):
            stem, ext = os.path.splitext(resolved)
            resolved = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"

        return resolved

    def _load_image_to_tensor(self, path: str, grayscale: bool = False):
        if not path or not os.path.exists(path):
            return None

        img = Image.open(path)
        if grayscale:
            img = img.convert("L")
            arr = np.array(img).astype(np.float32) / 255.0
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            img = img.convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0

        return torch.from_numpy(arr[np.newaxis, :, :, :])

    def _mesh_from_path(self, mesh_path: str):
        loaded = trimesh.load(mesh_path)
        if isinstance(loaded, trimesh.Scene):
            if len(loaded.geometry) == 1:
                return list(loaded.geometry.values())[0]
            return trimesh.util.concatenate(list(loaded.geometry.values()))
        return loaded

    def _stash_bake_maps(self, output_dir: str, output_path: str) -> dict[str, str | None]:
        base_stem = os.path.splitext(os.path.basename(output_path))[0]
        uid = uuid.uuid4().hex[:8]

        src_to_key = {
            "baked_diffuse.png": "albedo",
            "baked_normal.png": "normal",
            "baked_roughness.png": "roughness",
            "baked_metallic.png": "metallic",
            "baked_ao.png": "ao",
        }

        result = {k: None for k in src_to_key.values()}

        for src_name, key in src_to_key.items():
            src = os.path.join(output_dir, src_name)
            if not os.path.exists(src):
                continue
            dst = os.path.join(output_dir, f"{base_stem}_{src_name[:-4]}_{uid}.png")
            shutil.copy2(src, dst)
            result[key] = dst

        return result

    def remesh_rebake(
        self,
        file_path,
        target_triangles,
        output_path="",
        tri_tolerance=0.03,
        max_search_steps=10,
        bake_resolution=2048,
        bake_margin=40,
        samples=32,
        smooth_angle=75.0,
        weighted_normals=True,
        transfer_source_normals=False,
        remove_floaters=True,
        floater_min_triangles=384,
        floater_min_ratio=0.002,
        uv_method="smart",
        uv_angle=72.0,
        uv_seam_angle=65.0,
        uv_island_margin=0.003,
        auto_projection=True,
        ray_distance=0.005,
        cage_extrusion=0.0025,
        bake_normal=True,
        bake_roughness=True,
        bake_metallic=True,
        bake_ao=True,
        bake_emission=False,
        force_cpu=False,
        timeout_sec=7200,
        overwrite=True,
    ):
        script_path = os.path.join(os.path.dirname(__file__), "scripts", "blender_remesh_rebake_cli.py")
        if not os.path.exists(script_path):
            raise RuntimeError(f"Remesh script not found: {script_path}")

        blender_path, blender_version = check_blender()
        if not blender_path:
            raise RuntimeError("Blender not found. Install Blender in the environment to use this node.")

        input_path = self._resolve_existing_path(file_path)
        resolved_output_path = self._resolve_output_path(input_path, output_path, overwrite)

        cmd = [
            blender_path,
            "--background",
            "--python",
            script_path,
            "--",
            "--input", input_path,
            "--output", resolved_output_path,
            "--target-triangles", str(int(target_triangles)),
            "--tri-tolerance", str(float(tri_tolerance)),
            "--bake-resolution", str(int(bake_resolution)),
            "--bake-margin", str(int(bake_margin)),
            "--samples", str(int(samples)),
            "--max-search-steps", str(int(max_search_steps)),
            "--smooth-angle", str(float(smooth_angle)),
            "--floater-min-triangles", str(int(floater_min_triangles)),
            "--floater-min-ratio", str(float(floater_min_ratio)),
            "--uv-method", uv_method,
            "--uv-angle", str(float(uv_angle)),
            "--uv-seam-angle", str(float(uv_seam_angle)),
            "--uv-island-margin", str(float(uv_island_margin)),
        ]

        if not auto_projection:
            cmd.extend([
                "--ray-distance", str(float(ray_distance)),
                "--cage-extrusion", str(float(cage_extrusion)),
            ])

        cmd.append("--weighted-normals" if weighted_normals else "--no-weighted-normals")
        cmd.append("--transfer-source-normals" if transfer_source_normals else "--no-transfer-source-normals")
        cmd.append("--remove-floaters" if remove_floaters else "--no-remove-floaters")

        cmd.append("--bake-normal" if bake_normal else "--no-bake-normal")
        cmd.append("--bake-roughness" if bake_roughness else "--no-bake-roughness")
        cmd.append("--bake-metallic" if bake_metallic else "--no-bake-metallic")
        cmd.append("--bake-ao" if bake_ao else "--no-bake-ao")
        cmd.append("--bake-emission" if bake_emission else "--no-bake-emission")

        if force_cpu:
            cmd.append("--force-cpu")

        print(f"[BlenderRemeshRebake] Blender: {blender_version}")
        print("[BlenderRemeshRebake] Running:", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=int(timeout_sec),
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Blender remesh/rebake timed out after {timeout_sec}s") from e

        if result.stdout:
            # Keep logs concise but useful in Comfy console.
            for line in result.stdout.splitlines():
                if any(k in line for k in ["[", "DONE", "Best candidate", "bake projection", "floater removal", "Baking "]):
                    print(f"[BlenderRemeshRebake] {line}")

        if result.returncode != 0:
            raise RuntimeError(
                "Blender remesh/rebake failed\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        if not os.path.exists(resolved_output_path):
            raise RuntimeError(f"Output mesh not found after Blender run: {resolved_output_path}")

        output_mesh = self._mesh_from_path(resolved_output_path)
        output_dir = os.path.dirname(resolved_output_path)
        maps = self._stash_bake_maps(output_dir, resolved_output_path)

        albedo = self._load_image_to_tensor(maps["albedo"], grayscale=False)
        normal = self._load_image_to_tensor(maps["normal"], grayscale=False)
        roughness = self._load_image_to_tensor(maps["roughness"], grayscale=True)
        metallic = self._load_image_to_tensor(maps["metallic"], grayscale=True)
        ao = self._load_image_to_tensor(maps["ao"], grayscale=True)

        # Placeholders for missing maps
        if albedo is None:
            albedo = torch.zeros(1, 64, 64, 3)
        if normal is None:
            normal = torch.ones(1, 64, 64, 3) * torch.tensor([0.5, 0.5, 1.0])
        if roughness is None:
            roughness = torch.ones(1, 64, 64, 3) * 0.5
        if metallic is None:
            metallic = torch.zeros(1, 64, 64, 3)
        if ao is None:
            ao = torch.ones(1, 64, 64, 3)

        print(f"[BlenderRemeshRebake] Output: {resolved_output_path}")
        print(f"[BlenderRemeshRebake] Mesh: {len(output_mesh.vertices)} verts, {len(output_mesh.faces)} faces")

        return (output_mesh, resolved_output_path, albedo, normal, roughness, metallic, ao)
