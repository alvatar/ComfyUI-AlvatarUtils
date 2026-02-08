"""
Load GLB/GLTF/OBJ file from a path string and return as TRIMESH object.
"""

import os
import torch
import trimesh
import folder_paths

from .load_glb_to_trimesh import extract_textures_from_mesh


class LoadGLBFromPath:
    """
    Loads a GLB/GLTF/OBJ file from a path string and returns it as a TRIMESH object.
    Useful for chaining with other nodes that output file paths.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to mesh file (GLB/GLTF/OBJ/PLY/STL). Can be absolute or relative to input/output directory."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("trimesh", "albedo", "normal", "metallic_roughness", "occlusion")
    OUTPUT_TOOLTIPS = (
        "Loaded mesh as TRIMESH object.",
        "Albedo/base color texture (RGB).",
        "Normal map texture.",
        "Metallic-Roughness texture (glTF format: G=Roughness, B=Metallic).",
        "Occlusion texture (glTF format: R=Ambient Occlusion).",
    )
    FUNCTION = "load_mesh"
    CATEGORY = "Alvatar/3D"
    DESCRIPTION = "Loads a mesh file from a path string. Extracts all PBR textures in native glTF format."

    def load_mesh(self, file_path):
        """Load mesh file from path and return as trimesh object with all PBR textures."""
        # Resolve path
        resolved_path = None

        if os.path.isabs(file_path) and os.path.exists(file_path):
            resolved_path = file_path
        else:
            for base_dir in [folder_paths.get_input_directory(),
                             folder_paths.get_output_directory()]:
                test_path = os.path.join(base_dir, file_path)
                if os.path.exists(test_path):
                    resolved_path = test_path
                    break

        if not resolved_path:
            raise ValueError(f"Mesh file not found: {file_path}")

        print(f"[LoadGLBFromPath] Loading: {resolved_path}")

        # Load with trimesh
        scene_or_mesh = trimesh.load(resolved_path)

        # Extract textures
        textures = extract_textures_from_mesh(scene_or_mesh, resolved_path)

        # Get mesh
        mesh = scene_or_mesh
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 1:
                mesh = list(mesh.geometry.values())[0]
            else:
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        print(f"[LoadGLBFromPath] Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Report extracted textures
        for tex_name, tex in textures.items():
            if tex is not None:
                print(f"[LoadGLBFromPath] Extracted {tex_name}: {tex.shape}")

        # Create placeholders for missing textures
        albedo = textures['albedo']
        normal = textures['normal']
        metallic_roughness = textures['metallic_roughness']
        occlusion = textures['occlusion']

        if albedo is None:
            albedo = torch.zeros(1, 64, 64, 3)
        if normal is None:
            normal = torch.ones(1, 64, 64, 3) * torch.tensor([0.5, 0.5, 1.0])
        if metallic_roughness is None:
            metallic_roughness = torch.zeros(1, 64, 64, 3)
            metallic_roughness[..., 1] = 0.5
        if occlusion is None:
            occlusion = torch.ones(1, 64, 64, 3)

        return (mesh, albedo, normal, metallic_roughness, occlusion)
