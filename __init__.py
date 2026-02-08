"""
ComfyUI-AlvatarUtils
Unified plugin for image processing, texture processing, 3D mesh handling, and workflow utilities.

Categories:
- Alvatar/Image: BackgroundRemoval, PrepareImageFor3D, Upscale4x
- Alvatar/Texture: BlenderAOBaker, BlenderAOBakerFromTrimesh, BlenderRebakeTextures, MakeORM
- Alvatar/3D: LoadGLBToTrimesh, LoadGLBFromPath
- Alvatar/Utils: ResolvePath, DebugAny, Continue3
"""

__version__ = "2.4.0"
__author__ = "alvatar"

from .nodes import (
    # Image processing (Alvatar/Image)
    BackgroundRemoval,
    PrepareImageFor3D,
    Upscale4x,
    # Texture processing (Alvatar/Texture)
    BlenderAOBaker,
    BlenderAOBakerFromTrimesh,
    BlenderRebakeTextures,
    MakeORM,
    # 3D Mesh (Alvatar/3D)
    LoadGLBToTrimesh,
    LoadGLBFromPath,
    # Utilities (Alvatar/Utils)
    ResolvePath,
    DebugAny,
    Continue3,
)

# Node registration
NODE_CLASS_MAPPINGS = {
    # Image processing (Alvatar/Image)
    "BackgroundRemoval": BackgroundRemoval,
    "PrepareImageFor3D": PrepareImageFor3D,
    "Upscale4x": Upscale4x,
    # Texture processing (Alvatar/Texture)
    "BlenderAOBaker": BlenderAOBaker,
    "BlenderAOBakerFromTrimesh": BlenderAOBakerFromTrimesh,
    "BlenderRebakeTextures": BlenderRebakeTextures,
    "MakeORM": MakeORM,
    # 3D Mesh (Alvatar/3D)
    "LoadGLBToTrimesh": LoadGLBToTrimesh,
    "LoadGLBFromPath": LoadGLBFromPath,
    # Utilities (Alvatar/Utils)
    "ResolvePath": ResolvePath,
    "DebugAny": DebugAny,
    "Continue3": Continue3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Image processing (Alvatar/Image)
    "BackgroundRemoval": "Background Removal",
    "PrepareImageFor3D": "Prepare Image for 3D",
    "Upscale4x": "Upscale 4x",
    # Texture processing (Alvatar/Texture)
    "BlenderAOBaker": "Blender AO Baker (Path)",
    "BlenderAOBakerFromTrimesh": "Blender AO Baker (Trimesh)",
    "BlenderRebakeTextures": "Blender Rebake Textures",
    "MakeORM": "Make ORM",
    # 3D Mesh (Alvatar/3D)
    "LoadGLBToTrimesh": "Load GLB to Trimesh",
    "LoadGLBFromPath": "Load GLB from Path",
    # Utilities (Alvatar/Utils)
    "ResolvePath": "Resolve Path",
    "DebugAny": "Debug Any",
    "Continue3": "Continue 3",
}

# Web directory for JavaScript extensions
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
