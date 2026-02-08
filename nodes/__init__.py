# Node imports from categorized subfolders
from .image import BackgroundRemoval, PrepareImageFor3D, Upscale4x
from .texture import BlenderAOBaker, BlenderAOBakerFromTrimesh, BlenderRebakeTextures, MakeORM
from .mesh import LoadGLBToTrimesh, LoadGLBFromPath
from .utils import ResolvePath, DebugAny, Continue3

__all__ = [
    # Image processing (Alvatar/Image)
    "BackgroundRemoval",
    "PrepareImageFor3D",
    "Upscale4x",
    # Textures (Alvatar/Texture)
    "BlenderAOBaker",
    "BlenderAOBakerFromTrimesh",
    "BlenderRebakeTextures",
    "MakeORM",
    # 3D Mesh (Alvatar/3D)
    "LoadGLBToTrimesh",
    "LoadGLBFromPath",
    # Utilities (Alvatar/Utils)
    "ResolvePath",
    "DebugAny",
    "Continue3",
]
