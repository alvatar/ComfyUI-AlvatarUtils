"""
Upscale - State-of-the-art image upscaling for 3D generation pipelines.

Uses the absolute best upscaling models available (2024-2025):
- DRCT-L: Best reconstruction quality (NTIRE 2024 Image Super-Resolution winner)
- HAT-GAN Sharp: Best perceptual/visual sharpness (GAN-enhanced hybrid attention)
- UltraSharp: Best for compressed/degraded images (trained on JPEG artifacts)

Designed for upscaling to 2048/4096 for Trellis2, etc.

Model Comparison:
┌─────────────────┬────────────┬───────────────────────────────────────────────────┐
│ Model           │ Best For   │ Technical Details                                 │
├─────────────────┼────────────┼───────────────────────────────────────────────────┤
│ DRCT-L          │ Clean      │ Dense Residual Connected Transformer. Winner of  │
│                 │ images     │ NTIRE 2024. Highest PSNR/SSIM scores. 27M params. │
│                 │            │ Best for already-clean images where you want      │
│                 │            │ mathematically accurate reconstruction.           │
├─────────────────┼────────────┼───────────────────────────────────────────────────┤
│ HAT-GAN Sharp   │ Perceptual │ Hybrid Attention Transformer with GAN training.   │
│                 │ quality    │ Produces visually sharper output with enhanced    │
│                 │            │ details. May add subtle texture. Best when you    │
│                 │            │ prioritize visual appeal over exact accuracy.     │
├─────────────────┼────────────┼───────────────────────────────────────────────────┤
│ UltraSharp      │ Degraded   │ Specialized for real-world degradations: JPEG     │
│                 │ images     │ compression, noise, blur. Restores detail from    │
│                 │            │ lossy sources. Best for web images, screenshots,  │
│                 │            │ or any image that's been compressed.              │
└─────────────────┴────────────┴───────────────────────────────────────────────────┘

Uses Spandrel for universal model loading (supports all major SR architectures).
"""

import os
import torch
import numpy as np
from PIL import Image

# Model configurations
MODELS = {
    "DRCT-L": {
        "filename": "4xNomos2_hq_drct-l.safetensors",
        "scale": 4,
        "description": "Best reconstruction quality - NTIRE 2024 winner",
        "url": "https://github.com/Phhofm/models/releases/download/4xNomos2_hq_drct-l/4xNomos2_hq_drct-l.safetensors"
    },
    "HAT-GAN Sharp": {
        "filename": "Real_HAT_GAN_SRx4_sharper.pth",
        "scale": 4,
        "description": "Best perceptual quality - sharpest visual output",
        "url": "https://github.com/XPixelGroup/HAT/releases/download/v1.0.0/Real_HAT_GAN_SRx4_sharper.pth"
    },
    "UltraSharp": {
        "filename": "4x-UltraSharp.pth",
        "scale": 4,
        "description": "Best for compressed/degraded images (JPEG artifacts)",
        "url": "https://github.com/Kim2091/models/releases/download/v1.0/4x-UltraSharp.pth"
    },
}

# Cache for loaded models
_model_cache = {}


def get_model_dir():
    """Get the upscale models directory."""
    # Try ComfyUI's model path first
    comfy_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models", "upscale_models")
    if os.path.exists(comfy_path):
        return os.path.abspath(comfy_path)

    # Fallback to container path
    container_path = "/app/comfyui/models/upscale_models"
    if os.path.exists(container_path):
        return container_path

    # Create in ComfyUI path if neither exists
    os.makedirs(comfy_path, exist_ok=True)
    return os.path.abspath(comfy_path)


def load_model(model_name, device="cuda"):
    """Load upscale model using Spandrel."""
    global _model_cache

    cache_key = f"{model_name}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        import spandrel
    except ImportError:
        raise ImportError(
            "Spandrel is required for upscaling. Install with: pip install spandrel"
        )

    model_config = MODELS[model_name]
    model_dir = get_model_dir()
    model_path = os.path.join(model_dir, model_config["filename"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Download from: {model_config['url']}\n"
            f"Place in: {model_dir}"
        )

    print(f"[Upscale] Loading {model_name} from {model_path}")

    # Load model with Spandrel (auto-detects architecture)
    model = spandrel.ModelLoader().load_from_file(model_path)
    model = model.to(device).eval()

    _model_cache[cache_key] = model
    print(f"[Upscale] {model_name} loaded successfully")

    return model


def tensor2pil(image_tensor):
    """Convert ComfyUI tensor (B,H,W,C) to PIL Image."""
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]
    img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    if img_np.shape[-1] == 4:
        return Image.fromarray(img_np, mode='RGBA')
    elif img_np.shape[-1] == 3:
        return Image.fromarray(img_np, mode='RGB')
    else:
        return Image.fromarray(img_np[:, :, 0], mode='L')


def pil2tensor(image):
    """Convert PIL Image to ComfyUI tensor (B,H,W,C)."""
    img_np = np.array(image).astype(np.float32) / 255.0
    if len(img_np.shape) == 2:
        img_np = img_np[:, :, np.newaxis]
    return torch.from_numpy(img_np).unsqueeze(0)


class Upscale4x:
    """
    State-of-the-art 4x image upscaling using 2024-2025's best models.

    All models output 4x resolution (e.g., 512×512 → 2048×2048).
    Use target_size_downsize to downsample after upscaling (supersampling for better quality).

    Model Selection Guide:
    ══════════════════════
    • DRCT-L (Recommended default)
      Best for: Clean renders, generated images, photos without compression
      Quality: Highest mathematical accuracy (PSNR/SSIM)
      Details: Preserves fine details faithfully. NTIRE 2024 competition winner.

    • HAT-GAN Sharp
      Best for: Artistic output, when you want maximum visual "pop"
      Quality: Highest perceptual sharpness (LPIPS)
      Details: GAN training adds subtle texture enhancement. May slightly
               alter colors/textures but looks visually sharper.

    • UltraSharp
      Best for: JPEG images, web downloads, screenshots, degraded sources
      Quality: Best degradation handling
      Details: Trained specifically to restore quality lost to compression.
               Can recover surprisingly well from heavily compressed images.

    VRAM Usage:
    ═══════════
    All models use ~2-4GB VRAM with default tile_size=512.
    For limited VRAM, reduce tile_size to 256 (slower but uses ~1GB).
    For maximum quality, increase tile_size to 1024 if VRAM allows.
    """

    MODEL_NAMES = list(MODELS.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (cls.MODEL_NAMES, {
                    "default": "DRCT-L",
                    "tooltip": (
                        "DRCT-L: Best mathematical reconstruction (clean images). "
                        "HAT-GAN Sharp: Best visual sharpness (perceptual quality). "
                        "UltraSharp: Best for JPEG/compressed images (degradation removal)."
                    )
                }),
            },
            "optional": {
                "downsize": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 64,
                    "tooltip": (
                        "Downsize to this resolution (longest edge) after 4x upscale. "
                        "0 = no downsize (output raw 4x). "
                        "Supersampling (4x then downsize) produces sharper results. "
                        "Common values: 1024 (preview), 2048 (Trellis2/Hunyuan3D)."
                    )
                }),
                "tile_size": ("INT", {
                    "default": 256,
                    "min": 128,
                    "max": 1024,
                    "step": 64,
                    "tooltip": (
                        "Processing tile size. Affects VRAM usage vs speed tradeoff. "
                        "512 (default): ~2-4GB VRAM, good balance. "
                        "256: ~1GB VRAM, slower, for limited GPU memory. "
                        "1024: Faster but needs ~6-8GB VRAM."
                    )
                }),
                "tile_overlap": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 128,
                    "step": 8,
                    "tooltip": (
                        "Pixel overlap between tiles to prevent seams. "
                        "32 (default): Good for most cases. "
                        "64+: Use if you notice tile boundaries in output. "
                        "Higher values = smoother blending but slower."
                    )
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "Alvatar/Image"
    DESCRIPTION = (
        "4x upscaling with SOTA models (2024-2025). "
        "DRCT-L: Best accuracy for clean images. "
        "HAT-GAN: Sharpest perceptual output. "
        "UltraSharp: Best for JPEG/compressed. "
        "Use downsize to supersample then reduce for sharper output."
    )

    def upscale(self, image, model="DRCT-L", downsize=0, tile_size=256, tile_overlap=32):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        upscale_model = load_model(model, device)
        scale = MODELS[model]["scale"]

        # Process each image in batch
        results = []
        for i in range(image.shape[0]):
            single_image = image[i:i+1]

            # Convert to tensor format expected by model (B,C,H,W)
            img_tensor = single_image.permute(0, 3, 1, 2).to(device)

            # Handle alpha channel: models only support RGB (3 channels)
            has_alpha = img_tensor.shape[1] == 4
            if has_alpha:
                rgb_tensor = img_tensor[:, :3, :, :]
                alpha_tensor = img_tensor[:, 3:4, :, :]
            else:
                rgb_tensor = img_tensor

            # Upscale RGB with tiling
            with torch.no_grad():
                if tile_size > 0 and (rgb_tensor.shape[2] > tile_size or rgb_tensor.shape[3] > tile_size):
                    upscaled_rgb = self._tile_upscale(rgb_tensor, upscale_model, tile_size, tile_overlap, scale)
                else:
                    upscaled_rgb = upscale_model(rgb_tensor)

                # Upscale alpha separately using bicubic interpolation
                if has_alpha:
                    import torch.nn.functional as F
                    _, _, new_h, new_w = upscaled_rgb.shape
                    upscaled_alpha = F.interpolate(alpha_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
                    upscaled_alpha = torch.clamp(upscaled_alpha, 0, 1)
                    upscaled = torch.cat([upscaled_rgb, upscaled_alpha], dim=1)
                else:
                    upscaled = upscaled_rgb

            # Convert back to ComfyUI format (B,H,W,C)
            upscaled = upscaled.permute(0, 2, 3, 1).cpu()

            # Clamp values
            upscaled = torch.clamp(upscaled, 0, 1)

            results.append(upscaled)

        result = torch.cat(results, dim=0)

        # Downsize to target if specified (supersampling)
        if downsize > 0:
            result = self._resize_to_target(result, downsize)

        return (result,)

    def _tile_upscale(self, img_tensor, model, tile_size, overlap, scale):
        """Process image in tiles for VRAM efficiency."""
        _, c, h, w = img_tensor.shape

        # Calculate output size
        out_h, out_w = h * scale, w * scale

        # Create output tensor
        output = torch.zeros((1, c, out_h, out_w), device=img_tensor.device, dtype=img_tensor.dtype)
        weights = torch.zeros((1, 1, out_h, out_w), device=img_tensor.device, dtype=img_tensor.dtype)

        # Calculate tile positions
        stride = tile_size - overlap

        y_positions = list(range(0, h - tile_size + 1, stride))
        if y_positions[-1] + tile_size < h:
            y_positions.append(h - tile_size)

        x_positions = list(range(0, w - tile_size + 1, stride))
        if x_positions[-1] + tile_size < w:
            x_positions.append(w - tile_size)

        # Create blending weight (feathered edges)
        weight_tile = self._create_weight_tile(tile_size * scale, overlap * scale, img_tensor.device)

        for y in y_positions:
            for x in x_positions:

                # Extract tile
                tile = img_tensor[:, :, y:y+tile_size, x:x+tile_size]

                # Upscale tile
                upscaled_tile = model(tile)

                # Calculate output position
                out_y, out_x = y * scale, x * scale
                out_tile_size = tile_size * scale

                # Add to output with blending
                output[:, :, out_y:out_y+out_tile_size, out_x:out_x+out_tile_size] += upscaled_tile * weight_tile
                weights[:, :, out_y:out_y+out_tile_size, out_x:out_x+out_tile_size] += weight_tile

        # Normalize by weights
        output = output / (weights + 1e-8)

        return output

    def _create_weight_tile(self, tile_size, overlap, device):
        """Create a weight tile with feathered edges for seamless blending."""
        weight = torch.ones((1, 1, tile_size, tile_size), device=device)

        if overlap > 0:
            # Create linear ramp for feathering
            ramp = torch.linspace(0, 1, overlap, device=device)

            # Apply to all edges
            for i in range(overlap):
                weight[:, :, i, :] *= ramp[i]
                weight[:, :, -(i+1), :] *= ramp[i]
                weight[:, :, :, i] *= ramp[i]
                weight[:, :, :, -(i+1)] *= ramp[i]

        return weight

    def _resize_to_target(self, tensor, target_size):
        """Resize tensor so max dimension equals target_size (up or down)."""
        import torch.nn.functional as F

        _, h, w, _ = tensor.shape
        max_dim = max(h, w)

        if max_dim == target_size:
            return tensor

        scale = target_size / max_dim
        new_h, new_w = int(h * scale), int(w * scale)

        # Convert to (B,C,H,W) for interpolation
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = F.interpolate(tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
        tensor = tensor.permute(0, 2, 3, 1)

        return torch.clamp(tensor, 0, 1)
