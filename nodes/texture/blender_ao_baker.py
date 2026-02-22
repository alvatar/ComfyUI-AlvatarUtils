"""
Ambient Occlusion baking nodes using Blender Cycles.
"""

import os
import subprocess
import tempfile
import uuid
import numpy as np
import torch
from PIL import Image

import folder_paths

from .common import BLENDER_PATH, log


class BlenderAOBaker:
    """
    Bakes Ambient Occlusion texture from a 3D mesh file using Blender Cycles.
    Runs Blender headlessly - no GUI required.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to mesh file (GLB/OBJ/FBX/PLY/STL). From Trellis Save or file path."
                }),
                "resolution": ("INT", {
                    "default": 2048, "min": 256, "max": 8192, "step": 256,
                    "tooltip": "Output AO texture resolution in pixels."
                }),
                "samples": ("INT", {
                    "default": 64, "min": 1, "max": 512, "step": 1,
                    "tooltip": "Cycles render samples. More = cleaner but slower. 64 is usually enough for AO."
                }),
                "ao_distance": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1,
                    "tooltip": "Max distance for occlusion rays. Smaller = tighter shadows, larger = softer ambient shadows."
                }),
            },
            "optional": {
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU (CUDA) for Cycles rendering. Much faster than CPU."
                }),
                "margin": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Pixel margin around UV islands. Prevents seam artifacts."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("ao_image", "ao_path")
    OUTPUT_TOOLTIPS = (
        "Baked AO texture as IMAGE tensor. Connect to MakeORM or preview.",
        "File path to saved AO PNG in output directory."
    )
    FUNCTION = "bake_ao"
    CATEGORY = "Alvatar/Texture"
    DESCRIPTION = "Bakes Ambient Occlusion texture from a mesh file using Blender Cycles. Accepts GLB/OBJ/FBX paths from Trellis Save or other mesh export nodes."

    def bake_ao(self, mesh_path, resolution, samples, ao_distance, use_gpu=True, margin=16):
        """Bake AO texture using Blender headlessly"""

        if not BLENDER_PATH:
            raise RuntimeError("Blender not found. Please install Blender.")

        # Resolve path relative to ComfyUI directories if not absolute
        if mesh_path and not os.path.isabs(mesh_path):
            for base_dir in [folder_paths.get_output_directory(),
                            folder_paths.get_input_directory()]:
                resolved_path = os.path.join(base_dir, mesh_path)
                if os.path.exists(resolved_path):
                    mesh_path = resolved_path
                    break

        if not mesh_path or not os.path.exists(mesh_path):
            raise ValueError(f"Mesh file not found: {mesh_path}")

        log(f"Starting AO bake: {os.path.basename(mesh_path)}")
        log(f"Settings: {resolution}px, {samples} samples, distance={ao_distance}")

        # Create output path in ComfyUI output folder
        output_dir = folder_paths.get_output_directory()
        ao_filename = f"ao_bake_{uuid.uuid4().hex[:8]}.png"
        ao_output_path = os.path.join(output_dir, ao_filename)

        # Create the Blender Python script
        blender_script = self._create_bake_script(
            mesh_path=mesh_path,
            output_path=ao_output_path,
            resolution=resolution,
            samples=samples,
            ao_distance=ao_distance,
            use_gpu=use_gpu,
            margin=margin
        )

        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(blender_script)
            script_path = f.name

        try:
            # Run Blender headlessly
            cmd = [BLENDER_PATH, "--background", "--python", script_path]

            log("Running Blender...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            # Log Blender output for debugging
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if '[AO Baker]' in line or 'Error' in line or 'error' in line:
                        log(f"Blender: {line}")

            if result.returncode != 0:
                log(f"Blender STDERR: {result.stderr}")
                log(f"Blender STDOUT: {result.stdout}")
                raise RuntimeError(f"Blender baking failed: {result.stderr}")

            log(f"Baking complete: {ao_filename}")

        finally:
            # Clean up temp script
            if os.path.exists(script_path):
                os.unlink(script_path)

        # Load the baked image and convert to ComfyUI format
        if not os.path.exists(ao_output_path):
            raise RuntimeError(f"AO output file not created: {ao_output_path}")

        # Load image and convert to tensor format [B, H, W, C]
        img = Image.open(ao_output_path).convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array[np.newaxis, :, :, :])

        return (img_tensor, ao_output_path)

    def _create_bake_script(self, mesh_path, output_path, resolution, samples, ao_distance, use_gpu, margin):
        """Generate the Blender Python script for AO baking"""

        mesh_path_escaped = mesh_path.replace("\\", "\\\\").replace('"', '\\"')
        output_path_escaped = output_path.replace("\\", "\\\\").replace('"', '\\"')

        script = f'''
import bpy
import os
import sys

print("[AO Baker] Starting AO bake...")

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
mesh_path = "{mesh_path_escaped}"
print(f"[AO Baker] Importing: {{mesh_path}}")

ext = os.path.splitext(mesh_path)[1].lower()
if ext in ['.glb', '.gltf']:
    bpy.ops.import_scene.gltf(filepath=mesh_path)
elif ext == '.obj':
    bpy.ops.wm.obj_import(filepath=mesh_path)
elif ext == '.fbx':
    bpy.ops.import_scene.fbx(filepath=mesh_path)
elif ext == '.ply':
    bpy.ops.wm.ply_import(filepath=mesh_path)
elif ext == '.stl':
    bpy.ops.wm.stl_import(filepath=mesh_path)
else:
    print(f"[AO Baker] Unsupported format: {{ext}}")
    sys.exit(1)

# Find mesh objects
mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if not mesh_objects:
    print("[AO Baker] No mesh objects found!")
    sys.exit(1)

print(f"[AO Baker] Found {{len(mesh_objects)}} mesh object(s)")

# Apply all transforms (fixes GLTF Y-up to Z-up rotation causing UV mismatch)
bpy.ops.object.select_all(action='DESELECT')
for obj in mesh_objects:
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
print("[AO Baker] Applied transforms to mesh")

# Setup Cycles renderer
scene = bpy.context.scene
scene.render.engine = 'CYCLES'

# GPU setup
if {use_gpu}:
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        cuda_available = any(d.type == 'CUDA' for d in prefs.devices)
        if cuda_available:
            for device in prefs.devices:
                device.use = device.type == 'CUDA'
            scene.cycles.device = 'GPU'
            print("[AO Baker] Using GPU (CUDA)")
        else:
            scene.cycles.device = 'CPU'
            print("[AO Baker] CUDA not available, using CPU")
    except Exception as e:
        scene.cycles.device = 'CPU'
        print(f"[AO Baker] GPU setup failed, using CPU: {{e}}")
else:
    scene.cycles.device = 'CPU'
    print("[AO Baker] Using CPU")

scene.cycles.samples = {samples}

# Set AO distance
scene.world.light_settings.distance = {ao_distance}

# Create AO image
ao_image = bpy.data.images.new("AO_Bake", width={resolution}, height={resolution})

# Process each mesh object
for obj in mesh_objects:
    print(f"[AO Baker] Processing: {{obj.name}}")

    # Ensure object has UV map
    if not obj.data.uv_layers:
        print(f"[AO Baker] Creating UV map for {{obj.name}}")
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.select_set(False)

    # Create or get material
    if not obj.data.materials:
        mat = bpy.data.materials.new(name=f"{{obj.name}}_Material")
        obj.data.materials.append(mat)

    # Setup material nodes for baking
    for mat in obj.data.materials:
        if mat is None:
            continue
        mat.use_nodes = True
        nodes = mat.node_tree.nodes

        # Create image texture node for baking target
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.image = ao_image
        tex_node.name = "AO_Bake_Target"
        nodes.active = tex_node

# Select all mesh objects for baking
bpy.ops.object.select_all(action='DESELECT')
for obj in mesh_objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_objects[0]

# Bake AO
print("[AO Baker] Baking AO...")
bpy.ops.object.bake(
    type='AO',
    margin={margin},
    use_clear=True
)

# Save the image
output_path = "{output_path_escaped}"
ao_image.filepath_raw = output_path
ao_image.file_format = 'PNG'
ao_image.save()

print(f"[AO Baker] Saved AO map to: {{output_path}}")
print("[AO Baker] Done!")
'''
        return script


class BlenderAOBakerFromTrimesh:
    """
    Bakes Ambient Occlusion from a TRIMESH input (from other 3D nodes).
    Saves mesh temporarily, bakes AO, returns texture.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH", {
                    "tooltip": "Trimesh object from Hy3D, Trellis mesh output, or other 3D nodes that output TRIMESH."
                }),
                "resolution": ("INT", {
                    "default": 2048, "min": 256, "max": 8192, "step": 256,
                    "tooltip": "Output AO texture resolution in pixels."
                }),
                "samples": ("INT", {
                    "default": 64, "min": 1, "max": 512, "step": 1,
                    "tooltip": "Cycles render samples. More = cleaner but slower. 64 is usually enough for AO."
                }),
                "ao_distance": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1,
                    "tooltip": "Max distance for occlusion rays. Smaller = tighter shadows, larger = softer ambient shadows."
                }),
            },
            "optional": {
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU (CUDA) for Cycles rendering. Much faster than CPU."
                }),
                "margin": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Pixel margin around UV islands. Prevents seam artifacts."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "TRIMESH")
    RETURN_NAMES = ("ao_image", "ao_path", "trimesh")
    OUTPUT_TOOLTIPS = (
        "Baked AO texture as IMAGE tensor. Connect to MakeORM or preview.",
        "File path to saved AO PNG in output directory.",
        "Pass-through of input trimesh for chaining with other nodes."
    )
    FUNCTION = "bake_ao"
    CATEGORY = "Alvatar/Texture"
    DESCRIPTION = "Bakes Ambient Occlusion from a TRIMESH object using Blender Cycles. For nodes that output mesh objects directly (Hy3D, Trellis mesh output)."

    def bake_ao(self, trimesh, resolution, samples, ao_distance, use_gpu=True, margin=16):
        """Bake AO from trimesh object"""

        if not BLENDER_PATH:
            raise RuntimeError("Blender not found. Please install Blender.")

        log("Baking AO from TRIMESH input")

        # Save trimesh to temporary GLB file
        temp_dir = folder_paths.get_temp_directory()
        temp_mesh_path = os.path.join(temp_dir, f"temp_mesh_{uuid.uuid4().hex[:8]}.glb")

        # Export trimesh to GLB
        trimesh.export(temp_mesh_path)
        log(f"Exported trimesh to: {temp_mesh_path}")

        try:
            # Use the path-based baker
            baker = BlenderAOBaker()
            ao_image, ao_path = baker.bake_ao(
                mesh_path=temp_mesh_path,
                resolution=resolution,
                samples=samples,
                ao_distance=ao_distance,
                use_gpu=use_gpu,
                margin=margin
            )
        finally:
            # Clean up temp mesh file
            if os.path.exists(temp_mesh_path):
                os.unlink(temp_mesh_path)

        # Pass through the trimesh unchanged
        return (ao_image, ao_path, trimesh)
