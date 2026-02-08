"""
BackgroundRemoval - High-quality background removal for 3D pipelines

Supports RMBG-2.0 (BiRefNet) and BEN2 models with transformers 5.x compatibility.
Unlike the original RMBG plugin, this avoids device_map="auto" which causes
meta tensor errors in transformers 5.x.
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2

# Model cache to avoid reloading
_model_cache = {}


def tensor2pil_single(image_tensor):
    """Convert single ComfyUI tensor (H,W,C) to PIL Image."""
    img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    if img_np.shape[-1] == 4:
        return Image.fromarray(img_np, mode='RGBA')
    elif img_np.shape[-1] == 3:
        return Image.fromarray(img_np, mode='RGB')
    elif img_np.shape[-1] == 1:
        return Image.fromarray(img_np[:, :, 0], mode='L')
    else:
        raise ValueError(f"Unsupported channel count: {img_np.shape[-1]}")


def tensor2pil(image_tensor):
    """Convert ComfyUI tensor (B,H,W,C) to PIL Image (first image in batch)."""
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]
    return tensor2pil_single(image_tensor)


def pil2tensor(image):
    """Convert PIL Image to ComfyUI tensor (B,H,W,C)."""
    img_np = np.array(image).astype(np.float32) / 255.0
    if len(img_np.shape) == 2:
        img_np = img_np[:, :, np.newaxis]
    return torch.from_numpy(img_np).unsqueeze(0)


def mask2tensor(mask_pil):
    """Convert PIL mask (L mode) to ComfyUI mask tensor (B,H,W)."""
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    return torch.from_numpy(mask_np).unsqueeze(0)


def refine_foreground(image_pil, mask_pil):
    """
    Apply Fast Foreground Color Estimation to reduce color bleeding at edges.

    This algorithm improves edge quality by:
    1. Creating a binary mask with threshold
    2. Blurring edges with Gaussian filter
    3. Blending original and blurred masks in transition regions
    4. Multiplying RGB by the refined mask to reduce color fringing

    Useful for photos with color bleeding, but may not improve already-clean masks.
    """
    # Convert to numpy arrays
    image_np = np.array(image_pil.convert("RGB")).astype(np.float32) / 255.0
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0

    # Create binary mask with threshold
    thresh = 0.45
    mask_binary = (mask_np > thresh).astype(np.float32)

    # Blur edges
    edge_blur = cv2.GaussianBlur(mask_binary, (3, 3), 0)

    # Find transition regions (semi-transparent areas)
    transition_mask = np.logical_and(mask_np > 0.05, mask_np < 0.95)

    # Blend original and blurred masks in transition regions
    alpha = 0.85
    mask_refined = np.where(
        transition_mask,
        alpha * mask_np + (1 - alpha) * edge_blur,
        mask_binary
    )

    # Further refine edge regions
    edge_region = np.logical_and(mask_np > 0.2, mask_np < 0.8)
    mask_refined = np.where(edge_region, mask_refined * 0.98, mask_refined)

    # Apply refined mask to each RGB channel
    refined_rgb = np.zeros_like(image_np)
    for c in range(3):
        refined_rgb[:, :, c] = image_np[:, :, c] * mask_refined

    # Convert back to PIL
    refined_pil = Image.fromarray((refined_rgb * 255).astype(np.uint8), mode='RGB')
    return refined_pil


class BackgroundRemoval:
    """
    High-quality background removal using RMBG-2.0 or BEN2.

    - RMBG-2.0: BiRefNet architecture (0.2B params), best overall quality
    - BEN2: Background Erase Network v2, excellent edge handling

    Both models output a foreground image and an alpha mask.
    """

    MODELS = {
        "RMBG-2.0": {
            "repo": "briaai/RMBG-2.0",
            "type": "birefnet",
            "default_res": 1024,
            "description": "Best quality, BiRefNet architecture"
        },
        "BEN2": {
            "repo": "PramaLLC/BEN2",
            "type": "ben2",
            "default_res": 1024,
            "description": "Excellent edge handling, Confidence Guided Matting"
        }
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image(s) to process. Supports batch processing - all images "
                               "in the batch will be processed with the same settings. RGB or RGBA format."
                }),
                "model": (list(cls.MODELS.keys()), {
                    "default": "RMBG-2.0",
                    "tooltip": "Background removal model to use:\n"
                               "• RMBG-2.0: BiRefNet architecture with 0.2B parameters. Best overall "
                               "quality for general images. Trained on diverse datasets.\n"
                               "• BEN2: Background Erase Network v2 with Confidence Guided Matting. "
                               "Excels at fine details like hair and fur edges."
                }),
            },
            "optional": {
                "sensitivity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mask detection sensitivity (0.0-1.0).\n"
                               "• 1.0 (default): Standard detection, works well for most images.\n"
                               "• Higher values (closer to 1.0): More aggressive - keeps more pixels "
                               "as foreground. Use if parts of your subject are being cut off.\n"
                               "• Lower values (closer to 0.0): More conservative - removes more pixels. "
                               "Use if background is bleeding into the mask."
                }),
                "process_res": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Internal processing resolution (256-2048 pixels).\n"
                               "• 1024 (default): Optimal for RMBG-2.0, matches training resolution.\n"
                               "• Higher (1536-2048): Better quality for large/detailed images, but uses "
                               "more VRAM (~4x at 2048) and is slower.\n"
                               "• Lower (512-768): Faster processing, less VRAM, but may miss fine details.\n"
                               "The image is resized internally, then the mask is scaled back to original size."
                }),
                "mask_blur": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Gaussian blur radius for mask edges (0-64 pixels).\n"
                               "• 0 (default): Sharp mask edges - best for 3D workflows where you need "
                               "precise cutouts.\n"
                               "• 1-4: Subtle softening, can reduce jagged edges on low-res images.\n"
                               "• 8-16: Noticeable feathering, creates soft transitions.\n"
                               "• 32+: Heavy blur, only for special effects.\n"
                               "Note: For 3D texturing, sharp masks (0) are usually preferred."
                }),
                "mask_offset": ("INT", {
                    "default": 0,
                    "min": -64,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Expand or shrink the mask boundary (-64 to +64 pixels).\n"
                               "• 0 (default): Use mask as-is from the model.\n"
                               "• Positive (1-10): EXPAND mask outward - use if the model cuts too tight "
                               "and you're losing edge pixels of your subject.\n"
                               "• Negative (-1 to -10): SHRINK mask inward - use if background is bleeding "
                               "into the edges of your subject.\n"
                               "Each unit applies one iteration of morphological dilation/erosion."
                }),
                "invert_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask output (swap foreground and background).\n"
                               "• False (default): Normal operation - subject is white (1.0) in mask.\n"
                               "• True: Inverted - subject becomes black (0.0), background becomes white.\n"
                               "Useful for extracting backgrounds or creating cutout effects."
                }),
                "refine_foreground": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply Fast Foreground Color Estimation to reduce edge color bleeding.\n"
                               "• False (default): Use original image colors - best for clean renders.\n"
                               "• True: Refine RGB values at edges to reduce color fringing. This multiplies "
                               "edge pixels by a refined mask to remove background color contamination.\n"
                               "Most useful for photos with colored backgrounds bleeding into subject edges. "
                               "For clean 3D renders, this is usually unnecessary and may darken edges."
                }),
                "background_color": (["none", "white", "black"], {
                    "default": "none",
                    "tooltip": "Background color for the output image.\n"
                               "• none (default): Transparent background (RGBA output). Best for feeding "
                               "into 3D generation models like Trellis or UltraShape.\n"
                               "• white: Solid white background (RGB output). Some models prefer white BG.\n"
                               "• black: Solid black background (RGB output). Useful for debugging mask edges."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "Alvatar/Image"
    DESCRIPTION = (
        "Remove background from images using state-of-the-art segmentation models. "
        "Outputs the foreground image (with optional background color) and an alpha mask. "
        "Supports RMBG-2.0 (best quality) and BEN2 (best edges). "
        "Compatible with transformers 5.x."
    )

    def _load_model(self, model_name: str):
        """Load model with caching. Avoids device_map='auto' for transformers 5.x compat."""
        global _model_cache

        if model_name in _model_cache:
            return _model_cache[model_name]

        model_info = self.MODELS[model_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[BackgroundRemoval] Loading {model_name} from {model_info['repo']}...")

        if model_info["type"] == "birefnet":
            # RMBG-2.0 / BiRefNet - manually load to avoid transformers 5.x meta tensor issues
            # AutoModelForImageSegmentation.from_pretrained() causes "Tensor.item() cannot be
            # called on meta tensors" regardless of device_map or local_files_only settings.
            # Solution: Load model class and weights manually like the original RMBG plugin.
            import folder_paths
            import os
            import sys
            import importlib.util
            from safetensors.torch import load_file

            # Find local model path
            local_model_path = os.path.join(folder_paths.models_dir, model_info["repo"])

            if not os.path.exists(local_model_path):
                raise FileNotFoundError(
                    f"RMBG-2.0 model not found at {local_model_path}\n"
                    f"This is a gated model. Download with:\n"
                    f"  huggingface-cli login\n"
                    f"  huggingface-cli download briaai/RMBG-2.0 --local-dir {local_model_path}"
                )

            print(f"[BackgroundRemoval] Loading BiRefNet from: {local_model_path}")

            # Paths to model files
            config_path = os.path.join(local_model_path, "BiRefNet_config.py")
            model_path = os.path.join(local_model_path, "birefnet.py")
            weights_path = os.path.join(local_model_path, "model.safetensors")

            # Verify required files exist
            for path, name in [(config_path, "BiRefNet_config.py"), (model_path, "birefnet.py"), (weights_path, "model.safetensors")]:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Required file missing: {path}")

            # Fix relative import in birefnet.py (change "from .BiRefNet_config" to "from BiRefNet_config")
            with open(model_path, 'r', encoding='utf-8') as f:
                model_content = f.read()
            model_content = model_content.replace("from .BiRefNet_config", "from BiRefNet_config")

            # Load BiRefNet_config module
            config_spec = importlib.util.spec_from_file_location("BiRefNet_config", config_path)
            config_module = importlib.util.module_from_spec(config_spec)
            sys.modules["BiRefNet_config"] = config_module
            config_spec.loader.exec_module(config_module)

            # Load birefnet module (with patched import)
            birefnet_spec = importlib.util.spec_from_file_location("birefnet", model_path)
            birefnet_module = importlib.util.module_from_spec(birefnet_spec)
            sys.modules["birefnet"] = birefnet_module
            # Execute the patched content instead of the file directly
            exec(model_content, birefnet_module.__dict__)

            # Create model instance
            print(f"[BackgroundRemoval] Initializing BiRefNet model...")
            model = birefnet_module.BiRefNet(config_module.BiRefNetConfig())

            # Load weights from safetensors
            print(f"[BackgroundRemoval] Loading weights from {weights_path}...")
            state_dict = load_file(weights_path)
            model.load_state_dict(state_dict)

            model.eval()
            # Disable gradients for inference
            for param in model.parameters():
                param.requires_grad = False
            # Use FP16 for VRAM efficiency
            if device == "cuda":
                model = model.half()
                torch.set_float32_matmul_precision('high')
            model = model.to(device)

            _model_cache[model_name] = ("birefnet", model, device)

        elif model_info["type"] == "ben2":
            # BEN2 - uses its own ben2 package
            try:
                from ben2 import BEN_Base
            except ImportError:
                raise ImportError(
                    "BEN2 requires the ben2 package. Install with:\n"
                    "pip install git+https://github.com/PramaLLC/BEN2.git"
                )

            model = BEN_Base.from_pretrained(model_info["repo"])
            model.eval()
            # Disable gradients for inference
            for param in model.parameters():
                param.requires_grad = False
            # Use FP16 for VRAM efficiency
            if device == "cuda":
                torch.set_float32_matmul_precision('high')
            model = model.to(device)

            _model_cache[model_name] = ("ben2", model, device)

        print(f"[BackgroundRemoval] {model_name} loaded successfully")
        return _model_cache[model_name]

    def _process_birefnet(self, model, image_pil, device, process_res):
        """Process image with BiRefNet (RMBG-2.0)."""
        from torchvision import transforms

        original_size = image_pil.size

        # Use configurable processing resolution
        transform = transforms.Compose([
            transforms.Resize((process_res, process_res)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        input_tensor = transform(image_pil.convert("RGB")).unsqueeze(0).to(device)
        # Match model dtype (FP16 on CUDA)
        if device == "cuda":
            input_tensor = input_tensor.half()

        with torch.no_grad():
            output = model(input_tensor)[-1]  # Get last output (highest resolution)
            mask = torch.sigmoid(output[0, 0]).float()  # Sigmoid to [0,1], convert to FP32 for PIL

        # Resize mask back to original size
        mask_pil = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize(original_size, Image.Resampling.LANCZOS)

        return mask_pil

    def _process_ben2(self, model, image_pil, device, process_res):  # noqa: ARG002
        """Process image with BEN2."""
        # BEN2 has a simpler interface - device is managed internally
        # Resize for processing
        original_size = image_pil.size
        aspect_ratio = original_size[1] / original_size[0]
        new_w = process_res
        new_h = int(process_res * aspect_ratio)
        resized_image = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        with torch.no_grad():
            # BEN2's inference method returns RGBA image directly
            foreground = model.inference(resized_image)

        # Extract alpha channel as mask
        if foreground.mode == 'RGBA':
            mask_pil = foreground.split()[-1]  # Get alpha channel
        else:
            # Fallback: convert to grayscale
            mask_pil = foreground.convert('L')

        # Resize mask back to original size
        mask_pil = mask_pil.resize(original_size, Image.Resampling.LANCZOS)

        return mask_pil

    def _apply_mask_adjustments(self, mask_pil, sensitivity, mask_blur, mask_offset, invert_output):
        """Apply sensitivity, blur, offset, and inversion to the mask."""
        # Apply sensitivity adjustment
        if sensitivity != 1.0:
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            # Scale mask values: higher sensitivity = more aggressive (keeps more)
            mask_np = mask_np * (1 + (1 - sensitivity))
            mask_np = np.clip(mask_np, 0, 1)
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')

        # Apply Gaussian blur to edges
        if mask_blur > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=mask_blur))

        # Apply morphological offset (dilate/erode)
        if mask_offset != 0:
            if mask_offset > 0:
                # Dilate (expand mask)
                for _ in range(mask_offset):
                    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))
            else:
                # Erode (shrink mask)
                for _ in range(-mask_offset):
                    mask_pil = mask_pil.filter(ImageFilter.MinFilter(3))

        # Invert if requested
        if invert_output:
            mask_pil = Image.fromarray(255 - np.array(mask_pil), mode='L')

        return mask_pil

    def _apply_background(self, image_pil, mask_pil, background_color, do_refine_foreground):
        """Apply mask to image with specified background."""
        # Optionally refine foreground colors to reduce edge bleeding
        if do_refine_foreground:
            image_pil = refine_foreground(image_pil, mask_pil)

        image_rgba = image_pil.convert("RGBA")

        # Apply mask as alpha channel
        image_rgba.putalpha(mask_pil)

        if background_color == "none":
            return image_rgba

        # Create background
        bg_colors = {
            "white": (255, 255, 255, 255),
            "black": (0, 0, 0, 255)
        }
        bg = Image.new("RGBA", image_pil.size, bg_colors[background_color])

        # Composite foreground over background
        result = Image.alpha_composite(bg, image_rgba)
        return result.convert("RGB")

    def remove_background(
        self,
        image,
        model="RMBG-2.0",
        sensitivity=1.0,
        process_res=1024,
        mask_blur=0,
        mask_offset=0,
        invert_output=False,
        refine_foreground=False,
        background_color="none"
    ):
        """Remove background from image batch with full parameter control."""
        # Load model once for entire batch
        model_type, loaded_model, device = self._load_model(model)

        # Handle batch processing - image is (B,H,W,C)
        batch_size = image.shape[0]
        result_images = []
        result_masks = []

        for i in range(batch_size):
            # Convert single image to PIL
            image_pil = tensor2pil_single(image[i])

            # Process based on model type
            if model_type == "birefnet":
                mask_pil = self._process_birefnet(loaded_model, image_pil, device, process_res)
            elif model_type == "ben2":
                mask_pil = self._process_ben2(loaded_model, image_pil, device, process_res)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Apply mask adjustments (sensitivity, blur, offset, invert)
            mask_pil = self._apply_mask_adjustments(
                mask_pil, sensitivity, mask_blur, mask_offset, invert_output
            )

            # Apply background (with optional foreground refinement)
            result_pil = self._apply_background(
                image_pil, mask_pil, background_color, refine_foreground
            )

            # Convert to tensors and collect
            result_images.append(pil2tensor(result_pil))
            result_masks.append(mask2tensor(mask_pil))

        # Concatenate batch results
        result_tensor = torch.cat(result_images, dim=0)
        mask_tensor = torch.cat(result_masks, dim=0)

        return (result_tensor, mask_tensor)
