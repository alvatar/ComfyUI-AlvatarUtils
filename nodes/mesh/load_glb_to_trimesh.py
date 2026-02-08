"""
Mesh loading utilities - Load GLB/OBJ files to TRIMESH objects with texture extraction.

glTF 2.0 PBR Texture Layout:
- baseColorTexture → albedo
- normalTexture → normal map
- metallicRoughnessTexture → G=Roughness, B=Metalness (R unused)
- occlusionTexture → R=Ambient Occlusion
"""

import os
import numpy as np
import torch
import trimesh
import folder_paths


def _pil_to_tensor(img):
    """Convert PIL Image to ComfyUI tensor format [B, H, W, C]"""
    if img is None:
        return None

    # Convert to RGB if needed (handle RGBA, L, etc.)
    if img.mode == 'RGBA':
        # Keep alpha for potential use
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert to numpy then tensor
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr[np.newaxis, :, :, :])

    return tensor


def _extract_from_material(mat):
    """Extract all PBR textures from a trimesh material object."""
    textures = {
        'albedo': None,
        'normal': None,
        'metallic_roughness': None,
        'occlusion': None,
    }

    if mat is None:
        return textures

    # Albedo / Base Color
    if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
        textures['albedo'] = _pil_to_tensor(mat.baseColorTexture)
    elif hasattr(mat, 'image') and mat.image is not None:
        textures['albedo'] = _pil_to_tensor(mat.image)

    # Normal map
    if hasattr(mat, 'normalTexture') and mat.normalTexture is not None:
        textures['normal'] = _pil_to_tensor(mat.normalTexture)

    # Metallic-Roughness (glTF format: G=roughness, B=metallic)
    if hasattr(mat, 'metallicRoughnessTexture') and mat.metallicRoughnessTexture is not None:
        textures['metallic_roughness'] = _pil_to_tensor(mat.metallicRoughnessTexture)

    # Occlusion (glTF format: R channel = AO)
    if hasattr(mat, 'occlusionTexture') and mat.occlusionTexture is not None:
        textures['occlusion'] = _pil_to_tensor(mat.occlusionTexture)

    return textures


def extract_textures_from_mesh(mesh, _file_path=None):
    """
    Extract PBR textures from a trimesh object or Scene.
    Returns dict with textures in their native glTF format:
    - 'albedo': Base color texture
    - 'normal': Normal map
    - 'metallic_roughness': G=Roughness, B=Metallic (glTF native)
    - 'occlusion': R=Ambient Occlusion (glTF native)
    """
    textures = {
        'albedo': None,
        'normal': None,
        'metallic_roughness': None,
        'occlusion': None,
    }

    try:
        # Handle Scene objects
        if isinstance(mesh, trimesh.Scene):
            for _geom_name, geom in mesh.geometry.items():
                if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
                    mat = geom.visual.material
                    extracted = _extract_from_material(mat)
                    # Merge extracted textures (first one found wins)
                    for key, val in extracted.items():
                        if textures[key] is None and val is not None:
                            textures[key] = val
                    # If we found albedo, we probably have all textures from this material
                    if textures['albedo'] is not None:
                        break

        # Handle single mesh
        elif hasattr(mesh, 'visual'):
            visual = mesh.visual
            if hasattr(visual, 'material'):
                textures = _extract_from_material(visual.material)

            # Also check TextureVisuals directly
            if textures['albedo'] is None and hasattr(visual, 'image') and visual.image is not None:
                textures['albedo'] = _pil_to_tensor(visual.image)

    except Exception as e:
        print(f"[LoadMesh] Warning: Could not extract textures: {e}")

    return textures


class LoadGLBToTrimesh:
    """
    Loads a GLB/GLTF/OBJ file and returns it as a TRIMESH object.
    Extracts all embedded PBR textures in their native glTF format.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        mesh_extensions = ('.glb', '.gltf', '.obj', '.ply', '.stl')
        mesh_files = []

        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.lower().endswith(mesh_extensions):
                    mesh_files.append(f)

        if not mesh_files:
            mesh_files = ["no mesh files found"]

        return {
            "required": {
                "mesh_file": (sorted(mesh_files), {
                    "tooltip": "Select or upload a 3D mesh file. Supported: GLB, GLTF, OBJ, PLY, STL.",
                    "file_upload": True,
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("trimesh", "file_path", "albedo", "normal", "metallic_roughness", "occlusion")
    OUTPUT_TOOLTIPS = (
        "Loaded mesh as TRIMESH object for use with other 3D nodes.",
        "Full path to the loaded mesh file.",
        "Albedo/base color texture (RGB).",
        "Normal map texture.",
        "Metallic-Roughness texture (glTF format: G=Roughness, B=Metallic).",
        "Occlusion texture (glTF format: R=Ambient Occlusion).",
    )
    FUNCTION = "load_mesh"
    CATEGORY = "Alvatar/3D"
    DESCRIPTION = "Loads a 3D mesh file and extracts all PBR textures. Outputs textures in native glTF format."

    @classmethod
    def IS_CHANGED(cls, mesh_file):
        """Return file modification time to detect changes."""
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, mesh_file)
        if os.path.exists(file_path):
            return os.path.getmtime(file_path)
        return float('nan')

    @classmethod
    def VALIDATE_INPUTS(cls, mesh_file):
        """Validate that the mesh file exists."""
        if mesh_file == "no mesh files found":
            return "No mesh files found in input directory. Please upload a GLB/GLTF/OBJ file."
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, mesh_file)
        if not os.path.exists(file_path):
            return f"Mesh file not found: {mesh_file}"
        return True

    def load_mesh(self, mesh_file):
        """Load mesh file and return as trimesh object with all PBR textures."""
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, mesh_file)

        if not os.path.exists(file_path):
            raise ValueError(f"Mesh file not found: {file_path}")

        print(f"[LoadGLBToTrimesh] Loading: {mesh_file}")

        # Load with trimesh, keeping scene structure for texture access
        scene_or_mesh = trimesh.load(file_path)

        # Extract textures before potentially merging geometries
        textures = extract_textures_from_mesh(scene_or_mesh, file_path)

        # Get the mesh (merge if multiple geometries)
        mesh = scene_or_mesh
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 1:
                mesh = list(mesh.geometry.values())[0]
            else:
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        print(f"[LoadGLBToTrimesh] Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Report extracted textures
        for tex_name, tex in textures.items():
            if tex is not None:
                print(f"[LoadGLBToTrimesh] Extracted {tex_name}: {tex.shape}")
            else:
                print(f"[LoadGLBToTrimesh] No {tex_name} texture found")

        # Create placeholder tensors for missing textures
        albedo = textures['albedo']
        normal = textures['normal']
        metallic_roughness = textures['metallic_roughness']
        occlusion = textures['occlusion']

        if albedo is None:
            albedo = torch.zeros(1, 64, 64, 3)

        if normal is None:
            # Default normal map (pointing up: 0.5, 0.5, 1.0 in tangent space)
            normal = torch.ones(1, 64, 64, 3) * torch.tensor([0.5, 0.5, 1.0])

        if metallic_roughness is None:
            # Default: G=0.5 (medium roughness), B=0.0 (non-metallic)
            metallic_roughness = torch.zeros(1, 64, 64, 3)
            metallic_roughness[..., 1] = 0.5  # Green channel = roughness

        if occlusion is None:
            # Default: R=1.0 (no occlusion / fully lit)
            occlusion = torch.ones(1, 64, 64, 3)

        return (mesh, file_path, albedo, normal, metallic_roughness, occlusion)
