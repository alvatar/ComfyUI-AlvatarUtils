"""
GLTF simplify node using glTF-Transform CLI.

Wraps command equivalent to:
    gltf-transform simplify input.glb output.glb --ratio 0.5 --error 0.001

CLI options mapped to node knobs:
- --ratio
- --error
- --lock-border
- --vertex-layout

Intentionally omitted global CLI options (not meaningful for this node UI):
- --verbose (debug logging only)
- --config (experimental extension loading)
- --allow-net (this node resolves local filesystem inputs)

Execution backend:
- Prefer local `gltf-transform` binary (fast, offline-friendly)
- Fallback to `npx --yes @gltf-transform/cli` if local binary is unavailable
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid

import folder_paths

from .load_glb_from_path import LoadGLBFromPath


class GLTFSimplify:
    """
    Simplify glTF/GLB mesh using glTF-Transform (meshoptimizer).

    Notes:
    - This operation is lossy.
    - For best results on split geometry, run weld before simplify in external tooling.
    - By default, output is written to ComfyUI output directory.
    """

    VERTEX_LAYOUTS = ["interleaved", "separate"]
    RUNNERS = ["auto", "gltf-transform", "npx"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Input GLB/GLTF path. Absolute path or relative to input/output/temp directories."
                }),
                "ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Target ratio (0..1) of vertices to keep. 0.5 keeps ~50%."
                }),
                "error": ("FLOAT", {
                    "default": 0.001,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.0001,
                    "tooltip": "Maximum simplification error as fraction of mesh radius. Lower = higher quality."
                }),
                "lock_border": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Map to --lock-border. Locks topological border vertices during simplification."
                }),
            },
            "optional": {
                "output_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Output path. If empty, auto-generates in ComfyUI output directory."
                }),
                "vertex_layout": (cls.VERTEX_LAYOUTS, {
                    "default": "interleaved",
                    "tooltip": "Map to --vertex-layout (global glTF-Transform option)."
                }),
                "runner": (cls.RUNNERS, {
                    "default": "auto",
                    "tooltip": "Execution backend: auto prefers local gltf-transform, fallback npx."
                }),
                "timeout_sec": ("INT", {
                    "default": 600,
                    "min": 30,
                    "max": 7200,
                    "step": 30,
                    "tooltip": "Subprocess timeout in seconds."
                }),
                "overwrite": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If false and output exists, append a unique suffix instead of overwriting."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("trimesh", "output_path", "albedo", "normal", "metallic_roughness", "occlusion")
    OUTPUT_TOOLTIPS = (
        "Simplified mesh as TRIMESH.",
        "Path of simplified output glTF/GLB file.",
        "Extracted albedo texture (or placeholder).",
        "Extracted normal texture (or placeholder).",
        "Extracted metallic-roughness texture (or placeholder).",
        "Extracted occlusion texture (or placeholder).",
    )
    FUNCTION = "simplify"
    CATEGORY = "Alvatar/3D"
    DESCRIPTION = (
        "Simplify GLB/GLTF meshes using glTF-Transform simplify (meshoptimizer). "
        "Exposes all simplify-specific CLI knobs plus relevant global CLI options."
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
            test = os.path.join(base, path)
            if os.path.exists(test):
                return test

        raise ValueError(f"File not found: {path}")

    def _resolve_output_path(self, input_path: str, output_path: str, overwrite: bool) -> str:
        if output_path:
            if os.path.isabs(output_path):
                resolved = output_path
            else:
                resolved = os.path.join(folder_paths.get_output_directory(), output_path)
        else:
            out_dir = folder_paths.get_output_directory()
            base = os.path.basename(input_path)
            stem, ext = os.path.splitext(base)
            if ext.lower() not in [".glb", ".gltf"]:
                ext = ".glb"
            resolved = os.path.join(out_dir, f"{stem}_simplified{ext}")

        os.makedirs(os.path.dirname(resolved), exist_ok=True)

        if not overwrite and os.path.exists(resolved):
            stem, ext = os.path.splitext(resolved)
            resolved = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"

        return resolved

    def _resolve_runner(self, runner: str) -> list[str]:
        gltf_transform_bin = shutil.which("gltf-transform")
        npx_bin = shutil.which("npx")

        if runner == "gltf-transform":
            if not gltf_transform_bin:
                raise RuntimeError("runner='gltf-transform' selected but 'gltf-transform' binary not found in PATH")
            return [gltf_transform_bin]

        if runner == "npx":
            if not npx_bin:
                raise RuntimeError("runner='npx' selected but 'npx' binary not found in PATH")
            return [npx_bin, "--yes", "@gltf-transform/cli"]

        # auto
        if gltf_transform_bin:
            return [gltf_transform_bin]
        if npx_bin:
            return [npx_bin, "--yes", "@gltf-transform/cli"]

        raise RuntimeError(
            "Neither 'gltf-transform' nor 'npx' is available. "
            "Install glTF-Transform CLI globally (recommended) or Node.js+npx."
        )

    def _build_command(
        self,
        runner_prefix: list[str],
        input_path: str,
        output_path: str,
        ratio: float,
        error: float,
        lock_border: bool,
        vertex_layout: str,
    ) -> list[str]:
        cmd = [
            *runner_prefix,
            "simplify",
            input_path,
            output_path,
            "--ratio", str(ratio),
            "--error", str(error),
            "--lock-border", "true" if lock_border else "false",
            "--vertex-layout", vertex_layout,
        ]

        return cmd

    def simplify(
        self,
        file_path,
        ratio,
        error,
        lock_border,
        output_path="",
        vertex_layout="interleaved",
        runner="auto",
        timeout_sec=600,
        overwrite=True,
    ):
        input_path = self._resolve_existing_path(file_path)
        resolved_output_path = self._resolve_output_path(input_path, output_path, overwrite)

        runner_prefix = self._resolve_runner(runner)
        cmd = self._build_command(
            runner_prefix=runner_prefix,
            input_path=input_path,
            output_path=resolved_output_path,
            ratio=ratio,
            error=error,
            lock_border=lock_border,
            vertex_layout=vertex_layout,
        )

        print("[GLTFSimplify] Running:", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=int(timeout_sec),
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"gltf-transform simplify timed out after {timeout_sec}s") from e

        if result.returncode != 0:
            raise RuntimeError(
                "gltf-transform simplify failed\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        if not os.path.exists(resolved_output_path):
            raise RuntimeError(f"Simplify command succeeded but output not found: {resolved_output_path}")

        print(f"[GLTFSimplify] Simplified mesh written to: {resolved_output_path}")

        # Reuse existing loader to produce TRIMESH + extracted PBR textures.
        mesh, albedo, normal, metallic_roughness, occlusion = LoadGLBFromPath().load_mesh(resolved_output_path)

        return (mesh, resolved_output_path, albedo, normal, metallic_roughness, occlusion)
