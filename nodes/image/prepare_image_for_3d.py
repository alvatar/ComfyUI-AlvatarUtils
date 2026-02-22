"""
PrepareImageFor3D - Detect, center, and square-crop images for 3D generation.

Detects the foreground object using various methods, then centers it
within a square bounding box. Designed for Trellis2, etc.

Detection Method Comparison:
┌────────────────────┬──────────┬───────────────────────────────────────────────────┐
│ Method             │ Speed    │ Best For                                          │
├────────────────────┼──────────┼───────────────────────────────────────────────────┤
│ alpha              │ Instant  │ Images with transparency (PNG with alpha).        │
│                    │          │ Use after BackgroundRemoval node.                 │
├────────────────────┼──────────┼───────────────────────────────────────────────────┤
│ uniform_background │ Fast     │ Solid color backgrounds (white, black, gray).     │
│                    │          │ Samples corners to detect background color.       │
├────────────────────┼──────────┼───────────────────────────────────────────────────┤
│ threshold          │ Fast     │ High contrast images (dark object on light bg).   │
│                    │          │ Uses Otsu's automatic thresholding.               │
├────────────────────┼──────────┼───────────────────────────────────────────────────┤
│ edge               │ Fast     │ Objects with clear outlines/edges.                │
│                    │          │ Uses Canny edge detection + contour filling.      │
├────────────────────┼──────────┼───────────────────────────────────────────────────┤
│ grabcut            │ Slow     │ Complex images. OpenCV's GrabCut algorithm.       │
│                    │          │ Assumes object is centered, background at edges.  │
├────────────────────┼──────────┼───────────────────────────────────────────────────┤
│ auto               │ Varies   │ Unknown inputs. Tries: alpha → uniform → grabcut. │
│                    │          │ Falls back through methods until one succeeds.    │
└────────────────────┴──────────┴───────────────────────────────────────────────────┘

Typical Workflow:
    LoadImage → BackgroundRemoval → PrepareImageFor3D (alpha) → Trellis2/Hunyuan3D
"""

import numpy as np
import cv2
from PIL import Image


def tensor2pil(image_tensor):
    """Convert ComfyUI tensor (B,H,W,C) to PIL Image."""
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]
    img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    if img_np.shape[-1] == 4:
        return Image.fromarray(img_np, mode='RGBA')
    elif img_np.shape[-1] == 3:
        return Image.fromarray(img_np, mode='RGB')
    elif img_np.shape[-1] == 1:
        return Image.fromarray(img_np[:, :, 0], mode='L')
    else:
        raise ValueError(f"Unsupported channel count: {img_np.shape[-1]}")


def pil2tensor(image):
    """Convert PIL Image to ComfyUI tensor (B,H,W,C)."""
    import torch
    img_np = np.array(image).astype(np.float32) / 255.0
    if len(img_np.shape) == 2:
        img_np = img_np[:, :, np.newaxis]
    return torch.from_numpy(img_np).unsqueeze(0)


class PrepareImageFor3D:
    """
    Detect, center, and square-crop images for 3D generation pipelines.

    What This Node Does:
    ════════════════════
    1. Detects the foreground object using the selected method
    2. Finds the bounding box around the detected object
    3. Expands to a square (centered on object) with configurable margin
    4. Optionally resizes to exact output dimensions

    Why Square Cropping Matters:
    ════════════════════════════
    Most 3D generation models (Trellis2, etc.) expect:
    - Square input images (1:1 aspect ratio)
    - Object centered in frame
    - Consistent padding around object
    This node automates the manual process of "centering and cropping".

    Recommended Workflow:
    ═════════════════════
    LoadImage → BackgroundRemoval → PrepareImageFor3D → [Upscale] → 3D Model

    Detection Method Selection:
    ═══════════════════════════
    • auto (Default): Tries methods in order until success. Safe choice.
    • alpha: FASTEST. Use after BackgroundRemoval (image already has transparency).
    • uniform_background: For solid-color backgrounds (studio photos, renders).
    • threshold: For high-contrast images (silhouettes, dark-on-light).
    • edge: For images with clear object outlines.
    • grabcut: SMARTEST but slowest. For complex scenes. Assumes object centered.
    """

    DETECTION_METHODS = ["auto", "alpha", "uniform_background", "threshold", "edge", "grabcut"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_method": (cls.DETECTION_METHODS, {
                    "default": "auto",
                    "tooltip": (
                        "auto: Tries alpha→uniform→grabcut until success. "
                        "alpha: Use existing transparency (fastest, after BackgroundRemoval). "
                        "uniform_background: Solid color backgrounds (samples corners). "
                        "threshold: High contrast (Otsu's method). "
                        "edge: Clear outlines (Canny detection). "
                        "grabcut: Complex scenes (slower, assumes centered object)."
                    )
                }),
            },
            "optional": {
                "margin": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "tooltip": (
                        "Padding around detected object (fraction of object size). "
                        "0.1 (default): 10% padding - good for most 3D models. "
                        "0.0: Tight crop, object fills frame. "
                        "0.2-0.3: More breathing room, useful for models that add context."
                    )
                }),
                "output_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 64,
                    "tooltip": (
                        "Final square output size in pixels. 0 = keep natural crop size. "
                        "512: Quick previews. 1024: Standard quality. "
                        "2048: High quality (Trellis2, Hunyuan3D). "
                        "Note: For best quality, use Upscale node after this instead of output_size."
                    )
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "Alvatar/Image"
    DESCRIPTION = (
        "Detects object, centers it, and crops to square. "
        "Ideal prep for Trellis2. "
        "Use 'alpha' method after BackgroundRemoval for best results."
    )

    def process(self, image, detection_method="auto", margin=0.1, output_size=0):
        pil_image = tensor2pil(image)

        # Ensure we work with RGBA for consistent handling
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')

        img_np = np.array(pil_image)

        # Detect object and get mask
        mask = self._detect_object(img_np, detection_method)

        if mask is None:
            print("[PrepareImageFor3D] Warning: Detection failed, returning original image")
            return (pil2tensor(pil_image),)

        # Find bounding box from mask
        bbox = self._get_bounding_box(mask)

        if bbox is None:
            print("[PrepareImageFor3D] Warning: No object found, returning original image")
            return (pil2tensor(pil_image),)

        # Expand to square with margin, centered on object
        square_bbox = self._expand_to_square(bbox, margin)

        # Crop the image
        result = self._crop_with_padding(pil_image, square_bbox)

        # Resize if requested
        if output_size > 0:
            result = result.resize((output_size, output_size), Image.Resampling.LANCZOS)

        return (pil2tensor(result),)

    def _detect_object(self, img_np, method):
        """Detect foreground object and return binary mask."""

        if method == "auto":
            # Try methods in order of preference
            for try_method in ["alpha", "uniform_background", "grabcut"]:
                mask = self._detect_object(img_np, try_method)
                if mask is not None and np.any(mask):
                    return mask
            return None

        elif method == "alpha":
            return self._detect_alpha(img_np)

        elif method == "uniform_background":
            return self._detect_uniform_background(img_np)

        elif method == "threshold":
            return self._detect_threshold(img_np)

        elif method == "edge":
            return self._detect_edge(img_np)

        elif method == "grabcut":
            return self._detect_grabcut(img_np)

        else:
            print(f"[PrepareImageFor3D] Unknown method: {method}")
            return None

    def _detect_alpha(self, img_np):
        """Detect object from alpha channel."""
        if img_np.shape[-1] < 4:
            return None

        alpha = img_np[:, :, 3]

        # Check if alpha has meaningful variation (not all opaque or all transparent)
        if alpha.min() > 250 or alpha.max() < 5:
            return None

        # Threshold at 50% opacity
        mask = (alpha > 127).astype(np.uint8) * 255
        return mask

    def _detect_uniform_background(self, img_np):
        """Detect object by finding uniform background color from corners."""
        rgb = img_np[:, :, :3]
        h, w = rgb.shape[:2]

        # Sample corners (10% of image size)
        corner_size = max(10, min(h, w) // 10)

        corners = [
            rgb[:corner_size, :corner_size],           # top-left
            rgb[:corner_size, -corner_size:],          # top-right
            rgb[-corner_size:, :corner_size],          # bottom-left
            rgb[-corner_size:, -corner_size:],         # bottom-right
        ]

        # Calculate average background color
        corner_pixels = np.concatenate([c.reshape(-1, 3) for c in corners])
        bg_color = np.median(corner_pixels, axis=0)

        # Check if corners are uniform (low std = uniform background)
        corner_std = np.std(corner_pixels, axis=0).mean()
        if corner_std > 50:  # Too varied, not a uniform background
            return None

        # Calculate color distance from background
        diff = np.sqrt(np.sum((rgb.astype(float) - bg_color) ** 2, axis=2))

        # Threshold: pixels significantly different from background are foreground
        threshold = max(30, corner_std * 3)
        mask = (diff > threshold).astype(np.uint8) * 255

        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def _detect_threshold(self, img_np):
        """Detect object using Otsu's thresholding."""
        rgb = img_np[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Apply Otsu's thresholding
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # If most pixels are foreground, invert (assume object is darker/lighter than background)
        if np.mean(mask) > 127:
            mask = 255 - mask

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def _detect_edge(self, img_np):
        """Detect object using Canny edge detection."""
        rgb = img_np[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Fill holes - find contours and fill
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Create filled mask from contours
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, -1)  # -1 fills the contours

        return mask

    def _detect_grabcut(self, img_np):
        """Detect object using OpenCV GrabCut algorithm."""
        rgb = img_np[:, :, :3]
        h, w = rgb.shape[:2]

        # Initialize mask: assume edges are background, center is probable foreground
        mask = np.zeros((h, w), np.uint8)

        # Border region = definite background (GC_BGD = 0)
        border = max(10, min(h, w) // 10)

        # Everything starts as probable background (GC_PR_BGD = 2)
        mask[:, :] = cv2.GC_PR_BGD

        # Center region = probable foreground (GC_PR_FGD = 3)
        mask[border:-border, border:-border] = cv2.GC_PR_FGD

        # Definite background at edges
        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

        # Initialize models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Run GrabCut
        try:
            cv2.grabCut(rgb, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        except cv2.error as e:
            print(f"[PrepareImageFor3D] GrabCut failed: {e}")
            return None

        # Create binary mask (foreground = GC_FGD or GC_PR_FGD)
        binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        return binary_mask

    def _get_bounding_box(self, mask):
        """Get bounding box (x, y, w, h) from binary mask."""
        # Find non-zero pixels
        coords = cv2.findNonZero(mask)

        if coords is None:
            return None

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(coords)

        return (x, y, w, h)

    def _expand_to_square(self, bbox, margin):
        """Expand bounding box to square with margin, centered on object."""
        x, y, w, h = bbox

        # Calculate center of bounding box
        center_x = x + w / 2
        center_y = y + h / 2

        # Square size = max dimension + margin
        size = max(w, h)
        size_with_margin = int(size * (1 + margin * 2))

        # Calculate square bounds centered on object
        half_size = size_with_margin / 2

        left = int(center_x - half_size)
        top = int(center_y - half_size)
        right = int(center_x + half_size)
        bottom = int(center_y + half_size)

        return (left, top, right, bottom)

    def _crop_with_padding(self, pil_image, bbox):
        """Crop image to bbox, padding with transparency if bbox extends beyond image."""
        left, top, right, bottom = bbox
        img_w, img_h = pil_image.size

        # Calculate padding needed if bbox extends beyond image
        pad_left = max(0, -left)
        pad_top = max(0, -top)
        pad_right = max(0, right - img_w)
        pad_bottom = max(0, bottom - img_h)

        # Clamp bbox to image bounds
        crop_left = max(0, left)
        crop_top = max(0, top)
        crop_right = min(img_w, right)
        crop_bottom = min(img_h, bottom)

        # Crop the valid region
        cropped = pil_image.crop((crop_left, crop_top, crop_right, crop_bottom))

        # If padding needed, create larger canvas with transparency
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            new_w = right - left
            new_h = bottom - top
            canvas = Image.new('RGBA', (new_w, new_h), (0, 0, 0, 0))
            canvas.paste(cropped, (pad_left, pad_top))
            return canvas

        return cropped
