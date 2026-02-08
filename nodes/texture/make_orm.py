import torch


class MakeORM:
    """
    Creates an ORM (Occlusion/Roughness/Metalness) packed texture from grayscale inputs.

    Standard PBR channel packing:
    - Red: Ambient Occlusion (AO)
    - Green: Roughness
    - Blue: Metalness

    Handles both 3D [B,H,W] and 4D [B,H,W,C] input tensors automatically.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ao": ("IMAGE", {
                    "tooltip": "Ambient Occlusion texture (grayscale). Goes into RED channel. From AO Baker or Chord."
                }),
                "roughness": ("IMAGE", {
                    "tooltip": "Roughness texture (grayscale). Goes into GREEN channel. From Chord or manual input."
                }),
                "metalness": ("IMAGE", {
                    "tooltip": "Metalness texture (grayscale). Goes into BLUE channel. From Chord or manual input."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("orm",)
    OUTPUT_TOOLTIPS = ("Packed ORM texture: R=AO, G=Roughness, B=Metalness. Standard PBR format for game engines.",)
    FUNCTION = "make_orm"
    CATEGORY = "Alvatar/Texture"
    DESCRIPTION = "Packs 3 grayscale textures into ORM format (R=AO, G=Roughness, B=Metalness). Connect AO Baker output + Chord roughness/metalness."

    def _extract_single_channel(self, img, name):
        """Extract a single channel from an image, handling various formats."""
        print(f"[MakeORM] {name} input shape: {img.shape}")

        # Handle 3D tensor [B, H, W] -> [B, H, W, 1]
        if img.dim() == 3:
            img = img.unsqueeze(-1)
            print(f"[MakeORM] {name} expanded to: {img.shape}")

        # Now img is [B, H, W, C]
        # Take first channel only (in case it's RGB, we just need grayscale)
        if img.shape[-1] > 1:
            img = img[..., 0:1]
            print(f"[MakeORM] {name} extracted first channel: {img.shape}")

        return img

    def make_orm(self, ao, roughness, metalness):
        # Extract single channel from each input
        o = self._extract_single_channel(ao, "ao")
        r = self._extract_single_channel(roughness, "roughness")
        m = self._extract_single_channel(metalness, "metalness")

        # Ensure all tensors are on the same device (use first tensor's device as reference)
        device = o.device
        if r.device != device:
            r = r.to(device)
            print(f"[MakeORM] Moved roughness to {device}")
        if m.device != device:
            m = m.to(device)
            print(f"[MakeORM] Moved metalness to {device}")

        # Concatenate along channel dimension: [B, H, W, 1] * 3 -> [B, H, W, 3]
        orm = torch.cat([o, r, m], dim=-1)
        print(f"[MakeORM] Output shape: {orm.shape}, device: {orm.device}")

        return (orm,)
