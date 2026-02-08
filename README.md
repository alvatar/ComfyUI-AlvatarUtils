# ComfyUI-AlvatarUtils

Custom nodes for ComfyUI focused on 3D asset production pipelines: background removal, image preparation, 4x upscaling, GLB mesh loading with PBR texture extraction, Blender-based AO baking and texture rebaking, and workflow utilities.

## Nodes

### Image (Alvatar/Image)

#### Background Removal

High-quality background removal using RMBG-2.0 (BiRefNet) or BEN2. Returns a foreground image and alpha mask.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| image | IMAGE | — | Input image(s), supports batch |
| model | choice | RMBG-2.0 | RMBG-2.0 (best quality) or BEN2 (good edges) |
| sensitivity | FLOAT | 1.0 | Mask detection sensitivity (0.0–1.0) |
| process_res | INT | 1024 | Internal processing resolution (256–2048) |
| mask_blur | INT | 0 | Gaussian blur on mask edges (0–64) |
| mask_offset | INT | 0 | Expand/shrink mask boundary (-64 to 64) |
| refine_foreground | BOOL | False | Apply Fast Foreground Color Estimation |
| background_color | choice | none | none / white / black |

**Outputs:** `image` (IMAGE), `mask` (MASK)

#### Prepare Image for 3D

Detects the foreground object, centers it, and produces a square crop ready for 3D generation (Trellis2, Hunyuan3D, UltraShape, etc.).

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| image | IMAGE | — | Input image(s) |
| detection_method | choice | auto | auto, alpha, uniform_background, threshold, edge, grabcut |
| margin | FLOAT | 0.1 | Padding around detected object (0.0–0.5) |
| output_size | INT | 0 | Final square size in px (0 = keep natural) |

**Outputs:** `image` (IMAGE)

Detection methods: **alpha** (use existing transparency), **uniform_background** (solid-color backgrounds), **threshold** (Otsu), **edge** (Canny), **grabcut** (complex scenes, slowest), **auto** (tries alpha → uniform → grabcut).

#### Upscale 4x

4x super-resolution using SOTA models via Spandrel. Tile-based processing for VRAM efficiency.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| image | IMAGE | — | Input image(s) |
| model | choice | DRCT-L | DRCT-L (best reconstruction), HAT-GAN Sharp (sharpest), UltraSharp (best for JPEG) |
| downsize | INT | 0 | Downsize after 4x for supersampling (0 = no downsize) |
| tile_size | INT | 256 | Processing tile size (128–1024) |
| tile_overlap | INT | 32 | Overlap between tiles (8–128) |

**Outputs:** `image` (IMAGE)

---

### 3D Mesh (Alvatar/3D)

#### Load GLB to Trimesh

Loads a mesh file from ComfyUI's input directory and extracts all embedded PBR textures in glTF format.

| Input | Type | Description |
|-------|------|-------------|
| mesh_file | file dropdown | GLB, GLTF, OBJ, PLY, STL |

**Outputs:** `trimesh` (TRIMESH), `file_path` (STRING), `albedo` (IMAGE), `normal` (IMAGE), `metallic_roughness` (IMAGE), `occlusion` (IMAGE)

Texture channels follow glTF convention: metallic_roughness has G=Roughness, B=Metallic; occlusion has R=AO.

#### Load GLB from Path

Same as above but takes a path string instead of a file dropdown. Useful for chaining with nodes that output file paths.

| Input | Type | Description |
|-------|------|-------------|
| file_path | STRING | Absolute or relative path to mesh file |

**Outputs:** `trimesh` (TRIMESH), `albedo` (IMAGE), `normal` (IMAGE), `metallic_roughness` (IMAGE), `occlusion` (IMAGE)

---

### Texture (Alvatar/Texture)

#### Blender AO Baker (Path)

Bakes ambient occlusion from a mesh file using Blender Cycles (headless).

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| mesh_path | STRING | — | Path to mesh file |
| resolution | INT | 2048 | AO texture resolution (256–8192) |
| samples | INT | 64 | Cycles render samples (1–512) |
| ao_distance | FLOAT | 1.0 | Max occlusion ray distance (0.01–100) |
| use_gpu | BOOL | True | Use CUDA for rendering |
| margin | INT | 16 | UV island margin in pixels (0–64) |

**Outputs:** `ao_image` (IMAGE), `ao_path` (STRING)

#### Blender AO Baker (Trimesh)

Same baking as above but accepts a TRIMESH input and passes it through for chaining.

**Outputs:** `ao_image` (IMAGE), `ao_path` (STRING), `trimesh` (TRIMESH)

#### Blender Rebake Textures

Transfers PBR textures (albedo, normal, roughness, metallic) from a source mesh onto a target mesh with different topology. Uses Blender Cycles ray-casting. Does **not** transfer AO — bake that separately with BlenderAOBaker.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| source_trimesh | TRIMESH | — | Source mesh (has the textures) |
| target_trimesh | TRIMESH | — | Target mesh (receives the textures) |
| albedo | IMAGE | — | Albedo / base color texture |
| normal | IMAGE | (optional) | Normal map |
| metallic_roughness | IMAGE | (optional) | glTF format (G=Roughness, B=Metallic) |
| resolution | INT | 2048 | Output texture resolution (256–8192) |
| samples | INT | 64 | Cycles samples (1–256) |
| uv_method | choice | angle_based | angle_based, smart_project, lightmap |
| seam_angle | FLOAT | 66.0 | Seam angle threshold (10–89) |
| cage_extrusion | FLOAT | 0.05 | Cage inflation for baking (0.001–1.0) |
| max_ray_distance | FLOAT | 0.1 | Max ray travel distance (0.01–2.0) |
| margin | INT | 16 | Texture bleed margin (0–64) |
| use_gpu | BOOL | True | Use CUDA for rendering |
| debug_mode | BOOL | False | Save .blend file for inspection |

**Outputs:** `baked_albedo` (IMAGE), `baked_normal` (IMAGE), `baked_roughness` (IMAGE), `baked_metallic` (IMAGE), `glb_path` (STRING), `trimesh` (TRIMESH)

#### Make ORM

Packs three grayscale textures into a single ORM texture (R=AO, G=Roughness, B=Metallic). Standard PBR channel packing.

| Input | Type | Description |
|-------|------|-------------|
| ao | IMAGE | Ambient Occlusion → Red channel |
| roughness | IMAGE | Roughness → Green channel |
| metalness | IMAGE | Metalness → Blue channel |

**Outputs:** `orm` (IMAGE)

---

### Utilities (Alvatar/Utils)

#### Resolve Path

Resolves a relative path to absolute by searching output/, input/, and temp/ directories. Useful for connecting nodes that save files with nodes that load by path.

**Outputs:** `absolute_path` (STRING)

#### Debug Any

Accepts any input type, displays type info (shape, dtype, device, value preview) in the node UI, and passes the value through unchanged.

**Outputs:** `output` (*), `debug_text` (STRING)

#### Continue 3

Synchronization barrier with 3 passthrough channels (all optional). Forces all connected inputs to complete before any output is available. Useful for VRAM management and parallel branch synchronization.

**Outputs:** `output1` (*), `output2` (*), `output3` (*)

---

## Requirements

**Python packages** (beyond what ComfyUI provides):

```
trimesh
Pillow
opencv-python-headless
spandrel
safetensors
transformers
```

**System dependencies:**
- Blender 3.0+ (for AO baking and texture rebaking nodes)
- CUDA-capable GPU (optional, for GPU-accelerated baking and upscaling)

## Installation

Clone into ComfyUI's custom_nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alvatar/ComfyUI-AlvatarUtils.git
pip install -r ComfyUI-AlvatarUtils/requirements.txt
```

Restart ComfyUI. All nodes appear under the **Alvatar/** category.

## License

MIT
