# ComfyUI-AlvatarUtils

Custom nodes for ComfyUI focused on 3D asset production pipelines: background removal, image preparation, 4x upscaling, GLB mesh loading with PBR texture extraction, glTF mesh simplification, Blender-based remesh+rebake pipelines (target triangles + PBR maps), AO baking, and workflow utilities.

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

Loads a mesh file from ComfyUI's input directory and extracts all embedded PBR textures in glTF format. Supports direct browser upload via an in-node "Upload mesh" button.

| Input | Type | Description |
|-------|------|-------------|
| mesh_file | file dropdown | GLB, GLTF, OBJ, PLY, STL (recursive input subfolders; upload button available) |

**Outputs:** `trimesh` (TRIMESH), `file_path` (STRING), `albedo` (IMAGE), `normal` (IMAGE), `metallic_roughness` (IMAGE), `occlusion` (IMAGE)

Texture channels follow glTF convention: metallic_roughness has G=Roughness, B=Metallic; occlusion has R=AO.

#### Load GLB from Path

Same as above but takes a path string instead of a file dropdown. Useful for chaining with nodes that output file paths. Also supports direct browser upload via an in-node "Upload mesh" button.

| Input | Type | Description |
|-------|------|-------------|
| file_path | STRING | Absolute or relative path to mesh file (input/output/temp). If empty, uses mesh_file. |
| mesh_file | file dropdown (optional) | File picker from input dir (supports subfolders; upload button available). |

**Outputs:** `trimesh` (TRIMESH), `albedo` (IMAGE), `normal` (IMAGE), `metallic_roughness` (IMAGE), `occlusion` (IMAGE)

#### GLTF Simplify

Runs glTF-Transform simplify (meshoptimizer) from ComfyUI. Wraps:

```bash
gltf-transform simplify input.glb output.glb --ratio 0.5 --error 0.001
```

Execution backend: node prefers a local `gltf-transform` binary, with optional fallback to `npx @gltf-transform/cli`.

Omitted CLI globals: `--verbose`, `--config`, and `--allow-net` (not useful for this local-file node UI).

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| file_path | STRING | — | Input GLB/GLTF path (absolute or relative to input/output/temp) |
| ratio | FLOAT | 0.5 | `--ratio` target keep ratio (0..1) |
| error | FLOAT | 0.001 | `--error` max allowed simplification error |
| lock_border | BOOL | False | `--lock-border` preserve topological borders |
| output_path | STRING | auto | Output file path (auto in output/ if empty) |
| vertex_layout | choice | interleaved | `--vertex-layout` (`interleaved`/`separate`) |
| runner | choice | auto | `auto`, `gltf-transform`, or `npx` |
| timeout_sec | INT | 600 | Subprocess timeout |
| overwrite | BOOL | True | Overwrite output file or suffix uniquely |

**Outputs:** `trimesh` (TRIMESH), `output_path` (STRING), `albedo` (IMAGE), `normal` (IMAGE), `metallic_roughness` (IMAGE), `occlusion` (IMAGE)

#### Blender Remesh + Rebake

Runs a Blender headless pipeline to voxel-remesh toward a target triangle budget and rebake PBR textures.

The node wraps `nodes/mesh/scripts/blender_remesh_rebake_cli.py` and exposes the important quality knobs:

- triangle targeting (`target_triangles`, `tri_tolerance`, `max_search_steps`)
- geometry cleanup (`remove_floaters`, thresholds)
- UV strategy (`uv_method`, seam/angle/margin)
- shading (`smooth_angle`, weighted normals)
- projection control (`auto_projection` or manual ray/cage)
- map baking toggles (`normal`, `roughness`, `metallic`, `ao`, `emission`)

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| file_path | STRING | — | Input GLB/GLTF path (absolute or relative to input/output/temp) |
| target_triangles | INT | 250000 | Target triangle budget |
| output_path | STRING | auto | Output file path |
| tri_tolerance | FLOAT | 0.03 | Accepted triangle deviation (±3%) |
| max_search_steps | INT | 10 | Voxel search iterations |
| bake_resolution | INT | 2048 | Texture resolution |
| bake_margin | INT | 40 | Bake margin (pixels) |
| samples | INT | 32 | Cycles samples |
| smooth_angle | FLOAT | 75 | Smoothing angle |
| weighted_normals | BOOL | True | Apply weighted normals |
| transfer_source_normals | BOOL | False | Transfer source custom normals |
| remove_floaters | BOOL | True | Remove disconnected components |
| floater_min_triangles | INT | 384 | Floater min triangle threshold |
| floater_min_ratio | FLOAT | 0.002 | Floater min ratio threshold |
| uv_method | choice | smart | `smart` or `angle` |
| uv_angle | FLOAT | 72 | Smart UV angle |
| uv_seam_angle | FLOAT | 65 | Edge seam threshold for angle unwrap |
| uv_island_margin | FLOAT | 0.003 | UV island margin |
| auto_projection | BOOL | True | Auto derive ray/cage from mesh scale + voxel size |
| ray_distance | FLOAT | 0.005 | Manual ray distance (when auto off) |
| cage_extrusion | FLOAT | 0.0025 | Manual cage extrusion (when auto off) |
| bake_normal | BOOL | True | Bake normal map |
| bake_roughness | BOOL | True | Bake roughness map |
| bake_metallic | BOOL | True | Bake metallic map |
| bake_ao | BOOL | True | Bake AO map |
| bake_emission | BOOL | False | Bake emission map |
| force_cpu | BOOL | False | Force CPU rendering |
| timeout_sec | INT | 7200 | Blender timeout |
| overwrite | BOOL | True | Overwrite output file |

**Outputs:** `trimesh` (TRIMESH), `output_path` (STRING), `albedo` (IMAGE), `normal` (IMAGE), `roughness` (IMAGE), `metallic` (IMAGE), `ao` (IMAGE)

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

#### Continue

Synchronization barrier with dynamic passthrough channels. Starts with 2 inputs, and when both are connected a new input appears (`input3`), then `input4`, and so on. Forces all connected inputs to complete before downstream execution, useful for VRAM management and parallel branch synchronization.

**Outputs:** Dynamic `output1..N` passthroughs matching connected inputs.

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
- glTF-Transform CLI (`gltf-transform`) recommended for GLTF Simplify node
  - fallback: Node.js + `npx @gltf-transform/cli`
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
