"""
Texture rebaking node using Blender Cycles.
Transfers textures from a source mesh to a target mesh with different topology.

Rebakes: albedo, normal, roughness, metallic
Does NOT rebake: ambient occlusion (should be computed fresh on target mesh)
"""

import os
import subprocess
import tempfile
import uuid
import numpy as np
import torch
import trimesh
from PIL import Image

import folder_paths

from .common import BLENDER_PATH, log


class BlenderRebakeTextures:
    """
    Rebakes textures from a source mesh onto a target mesh with different topology.
    Uses Blender Cycles for ray-casting bake from source to target mesh.

    Transfers: albedo, normal, roughness, metallic
    Does NOT transfer: ambient occlusion (compute fresh with BlenderAOBaker instead)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_trimesh": ("TRIMESH", {
                    "tooltip": "Source mesh with UVs. The mesh that HAS the textures you want to transfer."
                }),
                "target_trimesh": ("TRIMESH", {
                    "tooltip": "Target mesh that will RECEIVE the baked textures. Can have different topology."
                }),
                "albedo": ("IMAGE", {
                    "tooltip": "Albedo/base color texture to transfer."
                }),
            },
            "optional": {
                "normal": ("IMAGE", {
                    "tooltip": "Normal map texture (optional)."
                }),
                "metallic_roughness": ("IMAGE", {
                    "tooltip": "Metallic-Roughness texture in glTF format: G=Roughness, B=Metallic (optional)."
                }),
                "resolution": ("INT", {
                    "default": 2048, "min": 256, "max": 8192, "step": 256,
                    "tooltip": "Output texture resolution in pixels."
                }),
                "samples": ("INT", {
                    "default": 64, "min": 1, "max": 256, "step": 1,
                    "tooltip": "Cycles samples per pixel. 64 recommended for quality."
                }),
                "uv_method": (["angle_based", "smart_project", "lightmap"], {
                    "default": "angle_based",
                    "tooltip": "UV unwrap method for target mesh."
                }),
                "seam_angle": ("FLOAT", {
                    "default": 66.0, "min": 10.0, "max": 89.0, "step": 1.0,
                    "tooltip": "Seam angle threshold for UV unwrapping."
                }),
                "cage_extrusion": ("FLOAT", {
                    "default": 0.05, "min": 0.001, "max": 1.0, "step": 0.01,
                    "tooltip": "Cage inflation distance for ray-casting."
                }),
                "max_ray_distance": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "Maximum ray travel distance."
                }),
                "margin": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Texture margin (bleed) in pixels."
                }),
                "margin_type": (["EXTEND", "ADJACENT_FACES"], {
                    "default": "EXTEND",
                    "tooltip": "Margin fill method."
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU (CUDA) for Cycles."
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save debug .blend file to output directory."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "TRIMESH")
    RETURN_NAMES = ("baked_albedo", "baked_normal", "baked_roughness", "baked_metallic", "glb_path", "trimesh")
    OUTPUT_TOOLTIPS = (
        "Baked albedo texture (RGB).",
        "Baked normal map (RGB).",
        "Baked roughness (grayscale).",
        "Baked metallic (grayscale).",
        "Path to the exported GLB file.",
        "Target mesh with new UVs.",
    )
    FUNCTION = "rebake"
    CATEGORY = "Alvatar/Texture"
    DESCRIPTION = "Transfers albedo, normal, roughness, metallic from source to target mesh. Does NOT transfer AO - use BlenderAOBaker for that."

    def _save_trimesh_to_temp(self, mesh, name, temp_dir):
        """Save trimesh to temp GLB file"""
        path = os.path.join(temp_dir, f"{name}_{uuid.uuid4().hex[:8]}.glb")
        mesh.export(path)
        return path

    def _save_texture_from_tensor(self, tensor, name, temp_dir):
        """Save a tensor image to a temp file"""
        if tensor is None:
            return None

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(-1).expand(-1, -1, -1, 3)

        img_array = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        path = os.path.join(temp_dir, f"{name}_{uuid.uuid4().hex[:8]}.png")
        img.save(path)
        return path

    def _load_image_to_tensor(self, path, grayscale=False):
        """Load image file and convert to ComfyUI tensor format"""
        if not path or not os.path.exists(path):
            return None

        img = Image.open(path)
        if grayscale:
            # Load as grayscale, then expand to RGB (R=G=B)
            img = img.convert('L')
            arr = np.array(img).astype(np.float32) / 255.0
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            img = img.convert('RGB')
            arr = np.array(img).astype(np.float32) / 255.0

        tensor = torch.from_numpy(arr[np.newaxis, :, :, :])
        return tensor

    def rebake(self, source_trimesh, target_trimesh, albedo,
               normal=None, metallic_roughness=None, resolution=2048, samples=64,
               uv_method="angle_based", seam_angle=66.0,
               cage_extrusion=0.05, max_ray_distance=0.1,
               margin=16, margin_type="EXTEND",
               use_gpu=True, debug_mode=False):
        """Rebake textures from source mesh to target mesh"""

        log("=" * 60)
        log("REBAKE TEXTURES")
        log("=" * 60)
        log(f"  Source: {len(source_trimesh.vertices)} verts, {len(source_trimesh.faces)} faces")
        log(f"  Target: {len(target_trimesh.vertices)} verts, {len(target_trimesh.faces)} faces")
        log(f"  Settings: {resolution}px, samples={samples}, uv={uv_method}")
        log("=" * 60)

        if not BLENDER_PATH:
            raise RuntimeError("Blender not found. Please install Blender.")

        temp_dir = folder_paths.get_temp_directory()
        output_dir = folder_paths.get_output_directory()

        # Save meshes to temp files
        source_mesh_path = self._save_trimesh_to_temp(source_trimesh, "source", temp_dir)
        target_mesh_path = self._save_trimesh_to_temp(target_trimesh, "target", temp_dir)

        # Save textures to temp files
        albedo_path = self._save_texture_from_tensor(albedo, "albedo", temp_dir)
        normal_path = self._save_texture_from_tensor(normal, "normal", temp_dir) if normal is not None else None
        metallic_roughness_path = self._save_texture_from_tensor(metallic_roughness, "metallic_roughness", temp_dir) if metallic_roughness is not None else None

        # Output paths
        uid = uuid.uuid4().hex[:8]
        output_path = os.path.join(output_dir, f"rebaked_mesh_{uid}.glb")
        baked_albedo_path = os.path.join(output_dir, f"baked_albedo_{uid}.png")
        baked_normal_path = os.path.join(output_dir, f"baked_normal_{uid}.png")
        baked_roughness_path = os.path.join(output_dir, f"baked_roughness_{uid}.png")
        baked_metallic_path = os.path.join(output_dir, f"baked_metallic_{uid}.png")

        # Create Blender script
        blender_script = self._create_bake_script(
            source_mesh_path=source_mesh_path,
            target_mesh_path=target_mesh_path,
            albedo_path=albedo_path,
            normal_path=normal_path,
            metallic_roughness_path=metallic_roughness_path,
            output_path=output_path,
            baked_albedo_path=baked_albedo_path,
            baked_normal_path=baked_normal_path if normal_path else None,
            baked_roughness_path=baked_roughness_path if metallic_roughness_path else None,
            baked_metallic_path=baked_metallic_path if metallic_roughness_path else None,
            resolution=resolution,
            samples=samples,
            uv_method=uv_method,
            seam_angle=seam_angle,
            cage_extrusion=cage_extrusion,
            max_ray_distance=max_ray_distance,
            margin=margin,
            margin_type=margin_type,
            use_gpu=use_gpu,
            debug_mode=debug_mode,
            output_dir=output_dir
        )

        # Write and run script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(blender_script)
            script_path = f.name

        try:
            cmd = [BLENDER_PATH, "--background", "--python", script_path]
            log("Running Blender...")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

            if result.stdout:
                for line in result.stdout.split('\n'):
                    if '[RebakeTextures]' in line or 'Error' in line:
                        log(f"Blender: {line}")

            if result.returncode != 0:
                log(f"Blender STDERR: {result.stderr}")
                raise RuntimeError(f"Texture rebaking failed: {result.stderr[:500]}")

            log(f"Rebaking complete!")

        finally:
            for p in [script_path, source_mesh_path, target_mesh_path, albedo_path, normal_path, metallic_roughness_path]:
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except:
                        pass

        # Load baked textures
        baked_albedo = self._load_image_to_tensor(baked_albedo_path)
        baked_normal = self._load_image_to_tensor(baked_normal_path) if normal_path else None
        baked_roughness = self._load_image_to_tensor(baked_roughness_path, grayscale=True) if metallic_roughness_path else None
        baked_metallic = self._load_image_to_tensor(baked_metallic_path, grayscale=True) if metallic_roughness_path else None

        # Placeholders for missing outputs
        if baked_albedo is None:
            baked_albedo = torch.zeros(1, resolution, resolution, 3)
        if baked_normal is None:
            baked_normal = torch.ones(1, resolution, resolution, 3) * torch.tensor([0.5, 0.5, 1.0])
        if baked_roughness is None:
            # Default roughness 0.5
            baked_roughness = torch.ones(1, resolution, resolution, 3) * 0.5
        if baked_metallic is None:
            # Default metallic 0.0
            baked_metallic = torch.zeros(1, resolution, resolution, 3)

        # Load output mesh
        if os.path.exists(output_path):
            output_mesh = trimesh.load(output_path)
        else:
            output_mesh = target_trimesh

        return (baked_albedo, baked_normal, baked_roughness, baked_metallic, output_path, output_mesh)

    def _create_bake_script(self, source_mesh_path, target_mesh_path,
                            albedo_path, normal_path, metallic_roughness_path, output_path,
                            baked_albedo_path, baked_normal_path, baked_roughness_path, baked_metallic_path,
                            resolution, samples, uv_method, seam_angle,
                            cage_extrusion, max_ray_distance, margin, margin_type,
                            use_gpu, debug_mode, output_dir):
        """Generate Blender Python script for texture rebaking"""

        def escape_path(p):
            if p is None:
                return "None"
            return '"' + p.replace("\\", "\\\\").replace('"', '\\"') + '"'

        script = f'''
import bpy
import os
import sys
import mathutils
import math

# Configuration
DEBUG_MODE = {debug_mode}
DEBUG_DIR = {escape_path(output_dir)}
UV_METHOD = "{uv_method}"
SEAM_ANGLE = {seam_angle}
SAMPLES = {samples}
CAGE_EXTRUSION = {cage_extrusion}
MAX_RAY_DISTANCE = {max_ray_distance}
MARGIN = {margin}
MARGIN_TYPE = "{margin_type}"
RESOLUTION = {resolution}
USE_GPU = {use_gpu}

# Output paths
BAKED_ALBEDO_PATH = {escape_path(baked_albedo_path)}
BAKED_NORMAL_PATH = {escape_path(baked_normal_path)}
BAKED_ROUGHNESS_PATH = {escape_path(baked_roughness_path)}
BAKED_METALLIC_PATH = {escape_path(baked_metallic_path)}

def debug_print(msg):
    print(f"[RebakeTextures] {{msg}}")

debug_print("Starting texture rebake...")

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import source mesh
source_path = {escape_path(source_mesh_path)}
debug_print(f"Importing source: {{source_path}}")
ext = os.path.splitext(source_path)[1].lower()
if ext in ['.glb', '.gltf']:
    bpy.ops.import_scene.gltf(filepath=source_path)
elif ext == '.obj':
    bpy.ops.wm.obj_import(filepath=source_path)

source_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
for i, obj in enumerate(source_objects):
    obj.name = f"source_{{i}}"
debug_print(f"Source: {{len(source_objects)}} meshes")

# Import target mesh
target_path = {escape_path(target_mesh_path)}
debug_print(f"Importing target: {{target_path}}")
ext = os.path.splitext(target_path)[1].lower()
if ext in ['.glb', '.gltf']:
    bpy.ops.import_scene.gltf(filepath=target_path)
elif ext == '.obj':
    bpy.ops.wm.obj_import(filepath=target_path)

all_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
target_objects = [obj for obj in all_objects if not obj.name.startswith("source_")]
for i, obj in enumerate(target_objects):
    obj.name = f"target_{{i}}"
debug_print(f"Target: {{len(target_objects)}} meshes")

if not source_objects or not target_objects:
    debug_print("ERROR: Missing source or target meshes!")
    sys.exit(1)

# Apply transforms
bpy.ops.object.select_all(action='DESELECT')
for obj in source_objects + target_objects:
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bpy.ops.object.select_all(action='DESELECT')

# Scale normalization
def get_mesh_bounds(objects):
    all_coords = []
    for obj in objects:
        bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        all_coords.extend(bbox)
    if not all_coords:
        return mathutils.Vector((0,0,0)), mathutils.Vector((0,0,0)), mathutils.Vector((0,0,0)), mathutils.Vector((0,0,0))
    min_co = mathutils.Vector((min(v.x for v in all_coords), min(v.y for v in all_coords), min(v.z for v in all_coords)))
    max_co = mathutils.Vector((max(v.x for v in all_coords), max(v.y for v in all_coords), max(v.z for v in all_coords)))
    return min_co, max_co, (min_co + max_co) / 2, max_co - min_co

src_min, src_max, src_center, src_size = get_mesh_bounds(source_objects)
tgt_min, tgt_max, tgt_center, tgt_size = get_mesh_bounds(target_objects)

src_max_dim = max(src_size.x, src_size.y, src_size.z)
tgt_max_dim = max(tgt_size.x, tgt_size.y, tgt_size.z)

if src_max_dim > 0.0001 and tgt_max_dim > 0.0001:
    scale_ratio = tgt_max_dim / src_max_dim
    if abs(scale_ratio - 1.0) > 0.001:
        for obj in target_objects:
            obj.scale = obj.scale * (1.0 / scale_ratio)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in target_objects:
            obj.select_set(True)
        bpy.ops.object.transform_apply(scale=True)
        bpy.ops.object.select_all(action='DESELECT')

# Center meshes
src_min, src_max, src_center, src_size = get_mesh_bounds(source_objects)
tgt_min, tgt_max, tgt_center, tgt_size = get_mesh_bounds(target_objects)
for obj in source_objects:
    obj.location = obj.location - src_center
for obj in target_objects:
    obj.location = obj.location - tgt_center
bpy.ops.object.select_all(action='DESELECT')
for obj in source_objects + target_objects:
    obj.select_set(True)
bpy.ops.object.transform_apply(location=True)
bpy.ops.object.select_all(action='DESELECT')

# Shrinkwrap
debug_print("Applying Shrinkwrap...")
if len(source_objects) > 1:
    bpy.ops.object.select_all(action='DESELECT')
    for obj in source_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = source_objects[0]
    bpy.ops.object.join()
    source_objects = [bpy.context.active_object]
    source_objects[0].name = "source_combined"

source_obj = source_objects[0]

for obj in target_objects:
    shrinkwrap = obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    shrinkwrap.target = source_obj
    shrinkwrap.wrap_method = 'NEAREST_SURFACEPOINT'
    shrinkwrap.wrap_mode = 'ON_SURFACE'
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

# Generate UVs for target
debug_print(f"Generating UVs (method={{UV_METHOD}})...")
for obj in target_objects:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    while obj.data.uv_layers:
        obj.data.uv_layers.remove(obj.data.uv_layers[0])
    obj.data.uv_layers.new(name='UVMap')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    if UV_METHOD == "angle_based":
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.edges_select_sharp(sharpness=math.radians(SEAM_ANGLE))
        bpy.ops.mesh.mark_seam(clear=False)
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    elif UV_METHOD == "smart_project":
        bpy.ops.uv.smart_project(angle_limit=SEAM_ANGLE, island_margin=0.02)
    elif UV_METHOD == "lightmap":
        bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)

# Load textures
albedo_path = {escape_path(albedo_path)}
normal_path = {escape_path(normal_path)}
metallic_roughness_path = {escape_path(metallic_roughness_path)}

textures = {{}}
if albedo_path and os.path.exists(albedo_path):
    img = bpy.data.images.load(albedo_path)
    img.colorspace_settings.name = 'sRGB'
    textures['albedo'] = img
    debug_print(f"Loaded albedo: {{img.size[0]}}x{{img.size[1]}}")

if normal_path and os.path.exists(normal_path):
    img = bpy.data.images.load(normal_path)
    img.colorspace_settings.name = 'Non-Color'
    textures['normal'] = img
    debug_print(f"Loaded normal: {{img.size[0]}}x{{img.size[1]}}")

if metallic_roughness_path and os.path.exists(metallic_roughness_path):
    img = bpy.data.images.load(metallic_roughness_path)
    img.colorspace_settings.name = 'Non-Color'
    textures['metallic_roughness'] = img
    debug_print(f"Loaded metallic_roughness: {{img.size[0]}}x{{img.size[1]}}")

# Setup source material with all textures
uv_layer_name = source_obj.data.uv_layers[0].name if source_obj.data.uv_layers else None

for obj in source_objects:
    mat = bpy.data.materials.new(name=f"{{obj.name}}_bake_mat")
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    uv_node = nodes.new('ShaderNodeUVMap')
    if uv_layer_name:
        uv_node.uv_map = uv_layer_name

    emission = nodes.new('ShaderNodeEmission')
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    if 'albedo' in textures:
        tex = nodes.new('ShaderNodeTexImage')
        tex.image = textures['albedo']
        tex.name = 'albedo_tex'
        links.new(uv_node.outputs['UV'], tex.inputs['Vector'])
        links.new(tex.outputs['Color'], emission.inputs['Color'])

    if 'normal' in textures:
        tex = nodes.new('ShaderNodeTexImage')
        tex.image = textures['normal']
        tex.name = 'normal_tex'
        tex.image.colorspace_settings.name = 'Non-Color'
        links.new(uv_node.outputs['UV'], tex.inputs['Vector'])

    if 'metallic_roughness' in textures:
        tex = nodes.new('ShaderNodeTexImage')
        tex.image = textures['metallic_roughness']
        tex.name = 'metallic_roughness_tex'
        tex.image.colorspace_settings.name = 'Non-Color'
        links.new(uv_node.outputs['UV'], tex.inputs['Vector'])

        # Separate channels for individual baking
        sep = nodes.new('ShaderNodeSeparateColor')
        sep.name = 'mr_separate'
        links.new(tex.outputs['Color'], sep.inputs['Color'])

        # Create grayscale outputs for roughness and metallic
        # Roughness = Green channel
        roughness_combine = nodes.new('ShaderNodeCombineColor')
        roughness_combine.name = 'roughness_tex'
        links.new(sep.outputs['Green'], roughness_combine.inputs['Red'])
        links.new(sep.outputs['Green'], roughness_combine.inputs['Green'])
        links.new(sep.outputs['Green'], roughness_combine.inputs['Blue'])

        # Metallic = Blue channel
        metallic_combine = nodes.new('ShaderNodeCombineColor')
        metallic_combine.name = 'metallic_tex'
        links.new(sep.outputs['Blue'], metallic_combine.inputs['Red'])
        links.new(sep.outputs['Blue'], metallic_combine.inputs['Green'])
        links.new(sep.outputs['Blue'], metallic_combine.inputs['Blue'])

# Create bake target images
bake_images = {{}}
bake_images['albedo'] = bpy.data.images.new("bake_albedo", width=RESOLUTION, height=RESOLUTION, alpha=False)
bake_images['albedo'].colorspace_settings.name = 'sRGB'

if 'normal' in textures:
    bake_images['normal'] = bpy.data.images.new("bake_normal", width=RESOLUTION, height=RESOLUTION, alpha=False)
    bake_images['normal'].colorspace_settings.name = 'Non-Color'

if 'metallic_roughness' in textures:
    bake_images['roughness'] = bpy.data.images.new("bake_roughness", width=RESOLUTION, height=RESOLUTION, alpha=False)
    bake_images['roughness'].colorspace_settings.name = 'Non-Color'
    bake_images['metallic'] = bpy.data.images.new("bake_metallic", width=RESOLUTION, height=RESOLUTION, alpha=False)
    bake_images['metallic'].colorspace_settings.name = 'Non-Color'

# Setup target material
for obj in target_objects:
    mat = bpy.data.materials.new(name=f"{{obj.name}}_target_mat")
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    emission = nodes.new('ShaderNodeEmission')
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    for name, img in bake_images.items():
        tex = nodes.new('ShaderNodeTexImage')
        tex.image = img
        tex.name = f"bake_{{name}}"

# Setup Cycles
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
if USE_GPU:
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type == 'CUDA'
        scene.cycles.device = 'GPU'
        debug_print("Using GPU")
    except:
        scene.cycles.device = 'CPU'
else:
    scene.cycles.device = 'CPU'

scene.cycles.samples = SAMPLES
scene.render.bake.use_selected_to_active = True
scene.render.bake.cage_extrusion = CAGE_EXTRUSION
scene.render.bake.max_ray_distance = MAX_RAY_DISTANCE
scene.render.bake.margin = MARGIN
scene.render.bake.margin_type = MARGIN_TYPE

# Bake function
def bake_texture(tex_name, source_tex_name=None):
    debug_print(f"Baking {{tex_name}}...")
    if source_tex_name:
        for obj in source_objects:
            mat = obj.data.materials[0]
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            emission = None
            tex_node = None
            for n in nodes:
                if n.type == 'EMISSION':
                    emission = n
                if n.name == source_tex_name:
                    tex_node = n
            if emission and tex_node:
                # Handle both texture nodes and combine nodes
                if hasattr(tex_node, 'outputs') and 'Color' in tex_node.outputs:
                    links.new(tex_node.outputs['Color'], emission.inputs['Color'])
                elif hasattr(tex_node, 'outputs') and len(tex_node.outputs) > 0:
                    links.new(tex_node.outputs[0], emission.inputs['Color'])

    bpy.ops.object.select_all(action='DESELECT')
    for obj in source_objects:
        obj.select_set(True)
        obj.hide_render = False

    for target_obj in target_objects:
        target_obj.select_set(True)
        bpy.context.view_layer.objects.active = target_obj
        for mat in target_obj.data.materials:
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.name == f"bake_{{tex_name}}":
                        mat.node_tree.nodes.active = node
                        break
        try:
            bpy.ops.object.bake(type='EMIT')
            debug_print(f"  Baked {{tex_name}}")
        except Exception as e:
            debug_print(f"  BAKE FAILED: {{e}}")
            return False
        target_obj.select_set(False)
    return True

# Bake all textures
if bake_texture('albedo', 'albedo_tex'):
    bake_images['albedo'].filepath_raw = BAKED_ALBEDO_PATH
    bake_images['albedo'].file_format = 'PNG'
    bake_images['albedo'].save()
    debug_print(f"Saved albedo: {{BAKED_ALBEDO_PATH}}")

if 'normal' in bake_images and BAKED_NORMAL_PATH:
    if bake_texture('normal', 'normal_tex'):
        bake_images['normal'].filepath_raw = BAKED_NORMAL_PATH
        bake_images['normal'].file_format = 'PNG'
        bake_images['normal'].save()
        debug_print(f"Saved normal: {{BAKED_NORMAL_PATH}}")

if 'roughness' in bake_images and BAKED_ROUGHNESS_PATH:
    if bake_texture('roughness', 'roughness_tex'):
        bake_images['roughness'].filepath_raw = BAKED_ROUGHNESS_PATH
        bake_images['roughness'].file_format = 'PNG'
        bake_images['roughness'].save()
        debug_print(f"Saved roughness: {{BAKED_ROUGHNESS_PATH}}")

if 'metallic' in bake_images and BAKED_METALLIC_PATH:
    if bake_texture('metallic', 'metallic_tex'):
        bake_images['metallic'].filepath_raw = BAKED_METALLIC_PATH
        bake_images['metallic'].file_format = 'PNG'
        bake_images['metallic'].save()
        debug_print(f"Saved metallic: {{BAKED_METALLIC_PATH}}")

# Save debug blend
if DEBUG_MODE:
    blend_path = os.path.join(DEBUG_DIR, "debug_rebake_scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    debug_print(f"Saved debug .blend: {{blend_path}}")

# Setup final material for GLB export
for obj in target_objects:
    mat = obj.data.materials[0]
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    if 'albedo' in bake_images:
        tex = nodes.new('ShaderNodeTexImage')
        tex.image = bake_images['albedo']
        links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])

    if 'normal' in bake_images:
        tex = nodes.new('ShaderNodeTexImage')
        tex.image = bake_images['normal']
        tex.image.colorspace_settings.name = 'Non-Color'
        normal_map = nodes.new('ShaderNodeNormalMap')
        links.new(tex.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])

    if 'roughness' in bake_images:
        tex = nodes.new('ShaderNodeTexImage')
        tex.image = bake_images['roughness']
        tex.image.colorspace_settings.name = 'Non-Color'
        links.new(tex.outputs['Color'], bsdf.inputs['Roughness'])

    if 'metallic' in bake_images:
        tex = nodes.new('ShaderNodeTexImage')
        tex.image = bake_images['metallic']
        tex.image.colorspace_settings.name = 'Non-Color'
        links.new(tex.outputs['Color'], bsdf.inputs['Metallic'])

# Hide source for export
for obj in source_objects:
    obj.hide_render = True
    obj.hide_viewport = True

# Export GLB
output_path = {escape_path(output_path)}
debug_print(f"Exporting GLB: {{output_path}}")

bpy.ops.object.select_all(action='DESELECT')
for obj in target_objects:
    obj.select_set(True)

bpy.ops.export_scene.gltf(
    filepath=output_path,
    use_selection=True,
    export_format='GLB',
    export_texcoords=True,
    export_normals=True,
    export_materials='EXPORT',
    export_image_format='AUTO'
)

debug_print("Done!")
'''
        return script
