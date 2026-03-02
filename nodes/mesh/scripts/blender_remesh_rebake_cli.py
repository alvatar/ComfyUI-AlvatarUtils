"""
Blender Script: Targeted Voxel Remesh + Texture Rebake
======================================================

Purpose
-------
Reduce a high-poly GLB to a target triangle budget using voxel remeshing,
then rebake textures from the original mesh onto the remeshed mesh.

CLI usage (headless)
--------------------
blender --background --python blender_remesh_rebake.py -- \
  --input /path/to/input.glb \
  --output /path/to/output.glb \
  --target-triangles 150000

Key required args
-----------------
- --input
- --output
- --target-triangles

Quality knobs
-------------
- --tri-tolerance         (default: 0.10 = ±10%)
- --bake-resolution       (default: 2048)
- --bake-margin           (default: 24 px)
- --samples               (default: 96)
- --ray-distance          (default: auto, scale-aware)
- --cage-extrusion        (default: auto, scale-aware)
- --max-search-steps      (default: 8)
- --bake-normal / --no-bake-normal
- --bake-roughness / --no-bake-roughness
- --bake-metallic / --no-bake-metallic
- --bake-ao / --no-bake-ao
- --bake-emission / --no-bake-emission
- --smooth-angle          (default: 70 degrees)
- --weighted-normals / --no-weighted-normals
- --transfer-source-normals / --no-transfer-source-normals
- --remove-floaters / --no-remove-floaters (default: remove)
- --floater-min-triangles (default: 256)
- --floater-min-ratio     (default: 0.0015 = 0.15% of lowpoly tris)
- --uv-method             (smart | angle, default: smart)
- --uv-seam-angle         (default: 65 degrees)
- --force-cpu

Notes
-----
- Blender 5.0.1+ recommended (5.0.0 has known baking issues).
- For very large meshes, voxel search may take time.
- Script keeps the best candidate found even if exact target range is not reached.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import deque
from dataclasses import dataclass

import bmesh
import bpy


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _argv_after_double_dash() -> list[str]:
    """Blender passes script args after `--`.

    Example:
      blender --background --python script.py -- --input a --output b
    """
    argv = sys.argv
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


@dataclass
class Config:
    input_glb: str
    output_glb: str
    target_triangles: int
    tri_tolerance: float
    bake_resolution: int
    bake_margin: int
    samples: int
    ray_distance: float | None
    cage_extrusion: float | None
    max_search_steps: int
    bake_normal: bool
    bake_roughness: bool
    bake_metallic: bool
    bake_ao: bool
    bake_emission: bool
    smooth_angle_deg: float
    weighted_normals: bool
    transfer_source_normals: bool
    remove_floaters: bool
    floater_min_triangles: int
    floater_min_ratio: float
    force_cpu: bool
    uv_method: str
    uv_angle_deg: float
    uv_seam_angle_deg: float
    uv_island_margin: float


def parse_cli() -> Config:
    parser = argparse.ArgumentParser(
        prog="blender_remesh_rebake.py",
        description="Targeted voxel remesh + texture rebake for GLB assets.",
    )

    parser.add_argument("--input", required=True, help="Input GLB file path")
    parser.add_argument("--output", required=True, help="Output GLB file path")
    parser.add_argument(
        "--target-triangles",
        required=True,
        type=int,
        help="Desired triangle budget (e.g. 100000..200000)",
    )

    parser.add_argument(
        "--tri-tolerance",
        type=float,
        default=0.10,
        help="Accepted relative deviation from target (default 0.10 = ±10%%)",
    )
    parser.add_argument("--bake-resolution", type=int, default=2048)
    parser.add_argument(
        "--bake-margin",
        type=int,
        default=24,
        help="Bake dilation margin in pixels to reduce seams",
    )
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument(
        "--ray-distance",
        type=float,
        default=None,
        help="Bake max ray distance (default: auto from model scale)",
    )
    parser.add_argument(
        "--cage-extrusion",
        type=float,
        default=None,
        help="Bake cage extrusion (default: auto from model scale)",
    )
    parser.add_argument("--max-search-steps", type=int, default=8)

    parser.add_argument("--bake-normal", dest="bake_normal", action="store_true")
    parser.add_argument("--no-bake-normal", dest="bake_normal", action="store_false")
    parser.set_defaults(bake_normal=True)

    parser.add_argument("--bake-roughness", dest="bake_roughness", action="store_true")
    parser.add_argument("--no-bake-roughness", dest="bake_roughness", action="store_false")
    parser.set_defaults(bake_roughness=True)

    parser.add_argument("--bake-metallic", dest="bake_metallic", action="store_true")
    parser.add_argument("--no-bake-metallic", dest="bake_metallic", action="store_false")
    parser.set_defaults(bake_metallic=True)

    parser.add_argument("--bake-ao", dest="bake_ao", action="store_true")
    parser.add_argument("--no-bake-ao", dest="bake_ao", action="store_false")
    parser.set_defaults(bake_ao=True)

    parser.add_argument("--bake-emission", dest="bake_emission", action="store_true")
    parser.add_argument("--no-bake-emission", dest="bake_emission", action="store_false")
    parser.set_defaults(bake_emission=False)

    parser.add_argument(
        "--smooth-angle",
        type=float,
        default=70.0,
        help="Normal smoothing angle in degrees (default 70)",
    )
    parser.add_argument("--weighted-normals", dest="weighted_normals", action="store_true")
    parser.add_argument("--no-weighted-normals", dest="weighted_normals", action="store_false")
    parser.set_defaults(weighted_normals=True)

    parser.add_argument(
        "--transfer-source-normals",
        dest="transfer_source_normals",
        action="store_true",
        help="Transfer custom split normals from highpoly to lowpoly",
    )
    parser.add_argument(
        "--no-transfer-source-normals",
        dest="transfer_source_normals",
        action="store_false",
    )
    parser.set_defaults(transfer_source_normals=False)

    parser.add_argument(
        "--remove-floaters",
        dest="remove_floaters",
        action="store_true",
        help="Remove small disconnected mesh islands after remesh",
    )
    parser.add_argument(
        "--no-remove-floaters",
        dest="remove_floaters",
        action="store_false",
        help="Keep all disconnected mesh islands",
    )
    parser.set_defaults(remove_floaters=True)

    parser.add_argument(
        "--floater-min-triangles",
        type=int,
        default=256,
        help="Minimum connected-component triangle count to keep",
    )
    parser.add_argument(
        "--floater-min-ratio",
        type=float,
        default=0.0015,
        help="Minimum connected-component ratio of lowpoly triangles to keep",
    )

    parser.add_argument("--force-cpu", action="store_true", default=False)
    parser.add_argument(
        "--uv-method",
        choices=["smart", "angle"],
        default="smart",
        help="UV unwrap strategy: smart projection or seam-based angle unwrap",
    )
    parser.add_argument("--uv-angle", type=float, default=66.0)
    parser.add_argument(
        "--uv-seam-angle",
        type=float,
        default=65.0,
        help="Sharp-edge angle used to mark seams when --uv-method=angle",
    )
    parser.add_argument("--uv-island-margin", type=float, default=0.005)

    args = parser.parse_args(_argv_after_double_dash())

    if args.target_triangles <= 1000:
        parser.error("--target-triangles must be > 1000")
    if args.bake_resolution <= 0:
        parser.error("--bake-resolution must be > 0")
    if args.bake_margin < 0:
        parser.error("--bake-margin must be >= 0")
    if not (0.0 <= args.tri_tolerance <= 0.9):
        parser.error("--tri-tolerance must be in [0.0, 0.9]")
    if not (0.0 <= args.smooth_angle <= 180.0):
        parser.error("--smooth-angle must be in [0.0, 180.0]")
    if args.ray_distance is not None and args.ray_distance < 0.0:
        parser.error("--ray-distance must be >= 0.0")
    if args.cage_extrusion is not None and args.cage_extrusion < 0.0:
        parser.error("--cage-extrusion must be >= 0.0")
    if not (0.0 <= args.uv_angle <= 180.0):
        parser.error("--uv-angle must be in [0.0, 180.0]")
    if not (0.0 <= args.uv_seam_angle <= 180.0):
        parser.error("--uv-seam-angle must be in [0.0, 180.0]")
    if not (0.0 <= args.uv_island_margin <= 1.0):
        parser.error("--uv-island-margin must be in [0.0, 1.0]")
    if args.floater_min_triangles < 0:
        parser.error("--floater-min-triangles must be >= 0")
    if not (0.0 <= args.floater_min_ratio <= 1.0):
        parser.error("--floater-min-ratio must be in [0.0, 1.0]")

    return Config(
        input_glb=os.path.abspath(args.input),
        output_glb=os.path.abspath(args.output),
        target_triangles=int(args.target_triangles),
        tri_tolerance=float(args.tri_tolerance),
        bake_resolution=int(args.bake_resolution),
        bake_margin=int(args.bake_margin),
        samples=int(args.samples),
        ray_distance=float(args.ray_distance) if args.ray_distance is not None else None,
        cage_extrusion=float(args.cage_extrusion) if args.cage_extrusion is not None else None,
        max_search_steps=int(args.max_search_steps),
        bake_normal=bool(args.bake_normal),
        bake_roughness=bool(args.bake_roughness),
        bake_metallic=bool(args.bake_metallic),
        bake_ao=bool(args.bake_ao),
        bake_emission=bool(args.bake_emission),
        smooth_angle_deg=float(args.smooth_angle),
        weighted_normals=bool(args.weighted_normals),
        transfer_source_normals=bool(args.transfer_source_normals),
        remove_floaters=bool(args.remove_floaters),
        floater_min_triangles=int(args.floater_min_triangles),
        floater_min_ratio=float(args.floater_min_ratio),
        force_cpu=bool(args.force_cpu),
        uv_method=str(args.uv_method),
        uv_angle_deg=float(args.uv_angle),
        uv_seam_angle_deg=float(args.uv_seam_angle),
        uv_island_margin=float(args.uv_island_margin),
    )


# -----------------------------------------------------------------------------
# Scene helpers
# -----------------------------------------------------------------------------


def ensure_object_mode():
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def clear_scene():
    ensure_object_mode()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # orphan cleanup
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in list(bpy.data.images):
        if block.users == 0:
            bpy.data.images.remove(block)


def import_glb(filepath: str):
    bpy.ops.import_scene.gltf(filepath=filepath)
    imported = list(bpy.context.selected_objects)
    print(f"Imported {len(imported)} objects from {filepath}")
    return imported


def join_mesh_objects(objects):
    meshes = [obj for obj in objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects were imported.")

    if len(meshes) == 1:
        return meshes[0]

    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    return bpy.context.active_object


def apply_rotation_scale(obj):
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)


def duplicate_object(obj, name: str):
    dup = obj.copy()
    dup.data = obj.data.copy()
    dup.name = name
    bpy.context.collection.objects.link(dup)
    return dup


def delete_object(obj):
    if obj is None:
        return
    mesh_data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh_data and mesh_data.users == 0:
        bpy.data.meshes.remove(mesh_data)


def triangle_count(obj) -> int:
    me = obj.data
    me.calc_loop_triangles()
    return len(me.loop_triangles)


def resolve_bake_projection_distances(
    source_obj,
    cfg: Config,
    voxel_hint: float | None = None,
) -> tuple[float, float]:
    """Resolve bake projection distances.

    Uses model scale + remesh voxel hint to reduce missed projection and avoid
    long-distance cross projection.
    """
    max_dim = max(source_obj.dimensions)

    # Base scale-aware defaults.
    auto_ray = max_dim * 0.0035
    auto_cage = max_dim * 0.0018

    # If we know remesh voxel size, tie projection to that geometric detail level.
    if voxel_hint is not None and voxel_hint > 0.0:
        auto_ray = max(auto_ray, voxel_hint * 0.9)
        auto_cage = max(auto_cage, voxel_hint * 0.45)

    auto_ray = min(max(auto_ray, 1e-4), 0.03)
    auto_cage = min(max(auto_cage, 5e-5), 0.015)

    ray = cfg.ray_distance if cfg.ray_distance is not None else auto_ray
    cage = cfg.cage_extrusion if cfg.cage_extrusion is not None else auto_cage

    if ray > 0.0 and cage > ray:
        cage = ray

    return ray, cage


def voxel_remesh(obj, voxel_size: float):
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    obj.data.remesh_voxel_size = float(voxel_size)

    # Version-safe optional toggles
    for attr, value in [
        ("use_remesh_fix_poles", True),
        ("use_remesh_smooth_normals", True),
        ("use_remesh_preserve_volume", True),
    ]:
        if hasattr(obj.data, attr):
            setattr(obj.data, attr, value)

    bpy.ops.object.voxel_remesh()


def unwrap_lowpoly_uv(
    obj,
    method: str,
    smart_angle_deg: float,
    seam_angle_deg: float,
    island_margin: float,
):
    """UV unwrap lowpoly mesh.

    - smart: Smart UV projection (robust fallback)
    - angle: seam-based unwrap from sharp edges (usually better continuity)
    """
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    if method == "smart":
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(smart_angle_deg),
            island_margin=island_margin,
        )
        print(f"  uv unwrap: smart (angle={smart_angle_deg:.1f}°, margin={island_margin})")
    else:
        # Seam-based unwrap for cleaner islands and fewer projection discontinuities.
        bpy.ops.mesh.select_mode(type="EDGE")
        bpy.ops.mesh.mark_seam(clear=True)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.edges_select_sharp(sharpness=math.radians(seam_angle_deg))
        bpy.ops.mesh.mark_seam(clear=False)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=island_margin)

        try:
            bpy.ops.uv.average_islands_scale()
        except Exception:
            pass

        try:
            bpy.ops.uv.pack_islands(rotate=True, margin=island_margin)
        except TypeError:
            bpy.ops.uv.pack_islands(margin=island_margin)

        print(
            f"  uv unwrap: angle (seam_angle={seam_angle_deg:.1f}°, margin={island_margin})"
        )

    bpy.ops.object.mode_set(mode="OBJECT")


def apply_normal_smoothing(obj, smooth_angle_deg: float, weighted_normals: bool):
    """Apply smooth shading + optional weighted normals.

    This is the key step to avoid faceted lighting on voxel-remeshed meshes.
    """
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    angle_rad = math.radians(smooth_angle_deg)

    # Blender 5+ preferred path
    applied_by_angle = False
    if hasattr(bpy.ops.object, "shade_smooth_by_angle"):
        try:
            bpy.ops.object.shade_smooth_by_angle(angle=angle_rad, keep_sharp_edges=True)
            applied_by_angle = True
            print(f"  smooth shading by angle: {smooth_angle_deg:.1f}°")
        except Exception as e:
            print(f"  shade_smooth_by_angle failed ({e}), falling back")

    # Fallback for older versions / edge cases
    if not applied_by_angle:
        bpy.ops.object.shade_smooth()
        for poly in obj.data.polygons:
            poly.use_smooth = True

        if hasattr(obj.data, "use_auto_smooth"):
            obj.data.use_auto_smooth = True
        if hasattr(obj.data, "auto_smooth_angle"):
            obj.data.auto_smooth_angle = angle_rad

        if hasattr(bpy.ops.object, "shade_auto_smooth"):
            try:
                bpy.ops.object.shade_auto_smooth(use_auto_smooth=True, angle=angle_rad)
            except Exception:
                pass

        print(f"  smooth shading fallback applied: {smooth_angle_deg:.1f}°")

    if weighted_normals:
        try:
            mod = obj.modifiers.new(name="WeightedNormals", type="WEIGHTED_NORMAL")
            if hasattr(mod, "keep_sharp"):
                mod.keep_sharp = True
            if hasattr(mod, "mode"):
                mode_items = {item.identifier for item in mod.bl_rna.properties["mode"].enum_items}
                if "FACE_AREA_WITH_ANGLE" in mode_items:
                    mod.mode = "FACE_AREA_WITH_ANGLE"
                elif "FACE_AREA" in mode_items:
                    mod.mode = "FACE_AREA"
            if hasattr(mod, "weight"):
                mod.weight = 50
            if hasattr(mod, "use_face_influence"):
                mod.use_face_influence = True

            bpy.ops.object.modifier_apply(modifier=mod.name)
            print("  weighted normals: enabled")
        except Exception as e:
            print(f"  weighted normals skipped ({e})")
    else:
        print("  weighted normals: disabled")


def transfer_source_custom_normals(source_obj, target_obj):
    """Transfer custom split normals from highpoly to lowpoly."""
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj

    try:
        mod = target_obj.modifiers.new(name="TransferSourceNormals", type="DATA_TRANSFER")
        mod.object = source_obj
        mod.use_loop_data = True
        mod.data_types_loops = {"CUSTOM_NORMAL"}

        if hasattr(mod, "loop_mapping"):
            mod.loop_mapping = "POLYINTERP_NEAREST"
        if hasattr(mod, "mix_mode"):
            mod.mix_mode = "REPLACE"
        if hasattr(mod, "mix_factor"):
            mod.mix_factor = 1.0

        bpy.ops.object.modifier_apply(modifier=mod.name)
        print("  source normal transfer: enabled")
    except Exception as e:
        print(f"  source normal transfer: skipped ({e})")


def remove_floating_components(obj, min_triangles: int, min_ratio: float):
    """Remove small disconnected mesh islands (floaters) from the lowpoly mesh.

    Keeps the largest connected component unconditionally to avoid deleting everything
    on aggressive thresholds.
    """
    mesh = obj.data
    polys = mesh.polygons
    face_count = len(polys)

    if face_count == 0:
        print("  floater removal: skipped (no faces)")
        return

    tri_weights = [max(1, len(poly.vertices) - 2) for poly in polys]
    total_tris = sum(tri_weights)
    ratio_cutoff = int(math.ceil(total_tris * max(0.0, min_ratio)))
    threshold = max(int(min_triangles), ratio_cutoff)

    if threshold <= 0:
        print("  floater removal: skipped (threshold <= 0)")
        return

    vert_to_polys: dict[int, list[int]] = {}
    for poly_idx, poly in enumerate(polys):
        for vid in poly.vertices:
            vert_to_polys.setdefault(vid, []).append(poly_idx)

    visited = bytearray(face_count)
    components: list[tuple[int, list[int]]] = []

    for start in range(face_count):
        if visited[start]:
            continue

        q = deque([start])
        visited[start] = 1
        comp_faces: list[int] = []
        comp_tris = 0

        while q:
            pid = q.popleft()
            comp_faces.append(pid)
            comp_tris += tri_weights[pid]

            for vid in polys[pid].vertices:
                for nbr in vert_to_polys.get(vid, []):
                    if not visited[nbr]:
                        visited[nbr] = 1
                        q.append(nbr)

        components.append((comp_tris, comp_faces))

    if len(components) <= 1:
        print("  floater removal: none found (single connected component)")
        return

    components.sort(key=lambda x: x[0], reverse=True)

    remove_face_indices: list[int] = []
    removed_components = 0
    removed_tris = 0

    for idx, (comp_tris, comp_faces) in enumerate(components):
        # Always keep the largest component as safety net.
        if idx == 0:
            continue
        if comp_tris < threshold:
            remove_face_indices.extend(comp_faces)
            removed_components += 1
            removed_tris += comp_tris

    if not remove_face_indices:
        print(
            f"  floater removal: no component below threshold {threshold} tris "
            f"(components={len(components)})"
        )
        return

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    faces_to_delete = [bm.faces[i] for i in remove_face_indices if i < len(bm.faces)]
    if faces_to_delete:
        bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    after_tris = triangle_count(obj)
    print(
        "  floater removal: "
        f"removed_components={removed_components}, removed_tris~={removed_tris}, "
        f"remaining_tris={after_tris}, threshold={threshold}"
    )


# -----------------------------------------------------------------------------
# Search for voxel size matching target triangles
# -----------------------------------------------------------------------------


@dataclass
class Candidate:
    obj: object
    voxel: float
    tris: int


def find_best_voxel_candidate(source_obj, cfg: Config) -> Candidate:
    target = cfg.target_triangles
    tri_min = int(target * (1.0 - cfg.tri_tolerance))
    tri_max = int(target * (1.0 + cfg.tri_tolerance))

    # Heuristic bounds from object size.
    max_dim = max(source_obj.dimensions)
    min_voxel = max(max_dim / 2048.0, 1e-5)  # fine
    max_voxel = max(max_dim / 20.0, min_voxel * 1.5)  # coarse
    start_voxel = max(max_dim / 120.0, min_voxel)
    start_voxel = min(start_voxel, max_voxel)

    print("\nVoxel search setup:")
    print(f"  target triangles: {target} (range {tri_min}..{tri_max})")
    print(f"  max dimension: {max_dim:.6f}")
    print(f"  voxel bounds: [{min_voxel:.6f}, {max_voxel:.6f}], start={start_voxel:.6f}")

    best: Candidate | None = None

    def evaluate(voxel: float, tag: str) -> tuple[int, object]:
        nonlocal best
        cand_obj = duplicate_object(source_obj, f"LowPoly_{tag}")
        voxel_remesh(cand_obj, voxel)
        tris = triangle_count(cand_obj)
        print(f"  trial {tag}: voxel={voxel:.6f}, tris={tris}")

        cand = Candidate(obj=cand_obj, voxel=voxel, tris=tris)

        if best is None or abs(cand.tris - target) < abs(best.tris - target):
            if best is not None and best.obj != source_obj:
                delete_object(best.obj)
            best = cand
        else:
            delete_object(cand_obj)

        return tris, cand_obj

    # Initial evaluation
    current_v = start_voxel
    current_t, _ = evaluate(current_v, "start")

    if tri_min <= current_t <= tri_max:
        assert best is not None
        print("  initial voxel already within target range")
        best.obj.name = "LowPoly"
        return best

    # Build bracket: low_v => too many tris, high_v => too few tris
    low_v = low_t = None
    high_v = high_t = None

    if current_t > tri_max:
        low_v, low_t = current_v, current_t
        v = current_v
        for i in range(max(2, cfg.max_search_steps)):
            v = min(v * 1.4, max_voxel)
            t, _ = evaluate(v, f"up{i}")
            if t <= tri_max:
                high_v, high_t = v, t
                break
            low_v, low_t = v, t
            if v >= max_voxel:
                break
    else:
        high_v, high_t = current_v, current_t
        v = current_v
        for i in range(max(2, cfg.max_search_steps)):
            v = max(v / 1.4, min_voxel)
            t, _ = evaluate(v, f"down{i}")
            if t >= tri_min:
                low_v, low_t = v, t
                break
            high_v, high_t = v, t
            if v <= min_voxel:
                break

    # Binary refinement if we got a bracket
    if low_v is not None and high_v is not None and low_v < high_v:
        for i in range(cfg.max_search_steps):
            mid_v = (low_v + high_v) * 0.5
            mid_t, _ = evaluate(mid_v, f"bin{i}")

            if tri_min <= mid_t <= tri_max:
                print("  reached target range during binary search")
                break

            if mid_t > tri_max:
                low_v, low_t = mid_v, mid_t
            else:
                high_v, high_t = mid_v, mid_t

    assert best is not None
    best.obj.name = "LowPoly"
    print(f"Best candidate: voxel={best.voxel:.6f}, tris={best.tris}")
    return best


# -----------------------------------------------------------------------------
# Baking
# -----------------------------------------------------------------------------


def setup_cycles(samples: int, force_cpu: bool):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples

    if force_cpu:
        scene.cycles.device = "CPU"
        print("Cycles device: CPU (forced)")
        return

    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        # Prefer CUDA when available; Blender may fallback internally.
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        any_gpu = False
        for d in prefs.devices:
            use = d.type in {"CUDA", "OPTIX", "HIP", "METAL", "ONEAPI"}
            d.use = use
            any_gpu = any_gpu or use
        scene.cycles.device = "GPU" if any_gpu else "CPU"
        print(f"Cycles device: {'GPU' if any_gpu else 'CPU'}")
    except Exception as e:
        scene.cycles.device = "CPU"
        print(f"Cycles GPU setup failed ({e}), falling back to CPU")


def create_target_material_with_image(obj, image, mat_name: str):
    """Create bake-target material.

    Important: the image node is active but NOT wired into shader inputs while baking.
    This avoids circular dependency artifacts.
    """
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (300, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)

    tex = nodes.new("ShaderNodeTexImage")
    tex.location = (-300, 0)
    tex.image = image

    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    tex.select = True
    nodes.active = tex

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    return mat, tex


def _first_node_of_type(nodes, node_type: str):
    for node in nodes:
        if node.type == node_type:
            return node
    return None


def _ensure_output_node(nodes):
    output = None
    for node in nodes:
        if node.type == "OUTPUT_MATERIAL":
            output = node
            if getattr(node, "is_active_output", False):
                break
    if output is None:
        output = nodes.new("ShaderNodeOutputMaterial")
    return output


def _socket_default_to_rgba(sock):
    if sock is None:
        return (0.0, 0.0, 0.0, 1.0)
    try:
        dv = sock.default_value
        if isinstance(dv, (float, int)):
            v = float(dv)
            return (v, v, v, 1.0)
        if hasattr(dv, "__len__"):
            if len(dv) >= 4:
                return (float(dv[0]), float(dv[1]), float(dv[2]), float(dv[3]))
            if len(dv) == 3:
                return (float(dv[0]), float(dv[1]), float(dv[2]), 1.0)
            if len(dv) == 1:
                v = float(dv[0])
                return (v, v, v, 1.0)
    except Exception:
        pass
    return (0.0, 0.0, 0.0, 1.0)


def _connect_socket_to_color(nodes, links, src_socket, dst_color_input):
    if src_socket is None:
        dst_color_input.default_value = (0.0, 0.0, 0.0, 1.0)
        return

    if src_socket.is_linked:
        from_socket = src_socket.links[0].from_socket
        source_type = getattr(from_socket, "type", "")

        if source_type == "RGBA":
            links.new(from_socket, dst_color_input)
            return

        if source_type == "VALUE":
            comb = nodes.new("ShaderNodeCombineColor")
            if hasattr(comb, "mode"):
                comb.mode = "RGB"
            links.new(from_socket, comb.inputs["Red"])
            links.new(from_socket, comb.inputs["Green"])
            links.new(from_socket, comb.inputs["Blue"])
            links.new(comb.outputs["Color"], dst_color_input)
            return

        if source_type == "VECTOR":
            sep = nodes.new("ShaderNodeSeparateXYZ")
            comb = nodes.new("ShaderNodeCombineColor")
            if hasattr(comb, "mode"):
                comb.mode = "RGB"
            links.new(from_socket, sep.inputs["Vector"])
            links.new(sep.outputs["X"], comb.inputs["Red"])
            links.new(sep.outputs["Y"], comb.inputs["Green"])
            links.new(sep.outputs["Z"], comb.inputs["Blue"])
            links.new(comb.outputs["Color"], dst_color_input)
            return

    dst_color_input.default_value = _socket_default_to_rgba(src_socket)


def _set_object_slot_materials(obj, materials):
    for idx, slot in enumerate(obj.material_slots):
        slot.material = materials[idx] if idx < len(materials) else None


def _build_emit_variant_material(source_mat, mode: str, idx: int):
    if source_mat is not None:
        mat = source_mat.copy()
        mat.name = f"{source_mat.name}__BAKE_{mode}_{idx}"
    else:
        mat = bpy.data.materials.new(name=f"BakeFallback_{mode}_{idx}")

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    output = _ensure_output_node(nodes)

    for link in list(links):
        if link.to_node == output and link.to_socket.name == "Surface":
            links.remove(link)

    bsdf = _first_node_of_type(nodes, "BSDF_PRINCIPLED")

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (output.location.x - 260, output.location.y)
    emission.inputs["Strength"].default_value = 1.0

    if mode == "METALLIC":
        if bsdf and "Metallic" in bsdf.inputs:
            _connect_socket_to_color(nodes, links, bsdf.inputs["Metallic"], emission.inputs["Color"])
        else:
            fallback = float(getattr(source_mat, "metallic", 0.0)) if source_mat is not None else 0.0
            emission.inputs["Color"].default_value = (fallback, fallback, fallback, 1.0)

    elif mode == "EMISSION":
        emission_color_socket = None
        emission_strength_socket = None
        if bsdf:
            emission_color_socket = bsdf.inputs.get("Emission Color")
            emission_strength_socket = bsdf.inputs.get("Emission Strength")

        _connect_socket_to_color(nodes, links, emission_color_socket, emission.inputs["Color"])

        if emission_strength_socket is not None and not emission_strength_socket.is_linked:
            try:
                emission.inputs["Strength"].default_value = float(emission_strength_socket.default_value)
            except Exception:
                emission.inputs["Strength"].default_value = 1.0

    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def _prepare_source_materials_for_emit_bake(source_obj, mode: str):
    original = [slot.material for slot in source_obj.material_slots]
    variants = [_build_emit_variant_material(mat, mode, idx) for idx, mat in enumerate(original)]
    _set_object_slot_materials(source_obj, variants)
    return original, variants


def _cleanup_temp_materials(materials):
    for mat in materials:
        try:
            if mat is not None and mat.users == 0:
                bpy.data.materials.remove(mat)
        except Exception:
            pass


def bake_map(
    source_obj,
    target_obj,
    bake_type: str,
    image,
    ray_distance: float,
    cage_extrusion: float,
    bake_margin: int,
):
    bake = bpy.context.scene.render.bake
    bake.use_selected_to_active = True
    bake.use_cage = False
    if hasattr(bake, "use_clear"):
        bake.use_clear = True
    if hasattr(bake, "margin"):
        bake.margin = bake_margin
    if hasattr(bake, "margin_type"):
        try:
            bake.margin_type = "EXTEND"
        except Exception:
            pass
    if hasattr(bake, "target"):
        try:
            bake.target = "IMAGE_TEXTURES"
        except Exception:
            pass

    if hasattr(bake, "cage_extrusion"):
        bake.cage_extrusion = cage_extrusion
    if hasattr(bake, "max_ray_distance"):
        bake.max_ray_distance = ray_distance

    bpy.ops.object.select_all(action="DESELECT")
    source_obj.select_set(True)
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj

    for mat in target_obj.data.materials:
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image == image:
                    node.select = True
                    mat.node_tree.nodes.active = node

    temp_original = None
    temp_variants = []

    if bake_type == "METALLIC":
        temp_original, temp_variants = _prepare_source_materials_for_emit_bake(source_obj, "METALLIC")
    elif bake_type == "EMISSION":
        temp_original, temp_variants = _prepare_source_materials_for_emit_bake(source_obj, "EMISSION")

    print(f"Baking {bake_type}...")

    try:
        if bake_type == "DIFFUSE":
            bake.use_pass_direct = False
            bake.use_pass_indirect = False
            bake.use_pass_color = True
            bpy.ops.object.bake(type="DIFFUSE")
        elif bake_type == "NORMAL":
            if hasattr(bake, "normal_space"):
                try:
                    bake.normal_space = "TANGENT"
                except Exception:
                    pass
            bpy.ops.object.bake(type="NORMAL")
        elif bake_type == "ROUGHNESS":
            bpy.ops.object.bake(type="ROUGHNESS")
        elif bake_type == "AO":
            bpy.ops.object.bake(type="AO")
        elif bake_type == "METALLIC":
            bpy.ops.object.bake(type="EMIT")
        elif bake_type == "EMISSION":
            bpy.ops.object.bake(type="EMIT")
        else:
            raise RuntimeError(f"Unsupported bake type: {bake_type}")
    finally:
        if temp_original is not None:
            _set_object_slot_materials(source_obj, temp_original)
            _cleanup_temp_materials(temp_variants)


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------


def export_glb(filepath: str, obj):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    try:
        bpy.ops.export_scene.gltf(filepath=filepath, export_format="GLB", use_selection=True)
    except TypeError:
        bpy.ops.export_scene.gltf(filepath=filepath, export_format="GLB")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    cfg = parse_cli()

    if not os.path.exists(cfg.input_glb):
        raise FileNotFoundError(f"Input not found: {cfg.input_glb}")

    version = bpy.app.version
    print("\n" + "=" * 72)
    print("TARGETED VOXEL REMESH + REBAKE")
    print("=" * 72)
    print(f"Blender version: {version[0]}.{version[1]}.{version[2]}")
    if version[0] >= 5 and version[1] == 0 and version[2] == 0:
        print("WARNING: Blender 5.0.0 has known baking issues; 5.0.1+ recommended.")

    print("\nConfig:")
    print(f"  input:            {cfg.input_glb}")
    print(f"  output:           {cfg.output_glb}")
    print(f"  target triangles: {cfg.target_triangles}")
    print(f"  tolerance:        ±{cfg.tri_tolerance * 100:.1f}%")
    print(f"  bake resolution:  {cfg.bake_resolution}")
    print(f"  bake margin:      {cfg.bake_margin}")
    print(f"  samples:          {cfg.samples}")
    print(f"  bake normal:      {cfg.bake_normal}")
    print(f"  bake roughness:   {cfg.bake_roughness}")
    print(f"  bake metallic:    {cfg.bake_metallic}")
    print(f"  bake ao:          {cfg.bake_ao}")
    print(f"  bake emission:    {cfg.bake_emission}")
    print(f"  smooth angle:     {cfg.smooth_angle_deg:.1f}°")
    print(f"  weighted normals: {cfg.weighted_normals}")
    print(f"  transfer normals: {cfg.transfer_source_normals}")
    print(f"  remove floaters:  {cfg.remove_floaters}")
    print(f"  floater min tris: {cfg.floater_min_triangles}")
    print(f"  floater min ratio:{cfg.floater_min_ratio:.6f}")
    print(f"  uv method:        {cfg.uv_method}")
    print(f"  uv seam angle:    {cfg.uv_seam_angle_deg:.1f}°")
    print(f"  uv angle:         {cfg.uv_angle_deg:.1f}°")
    print(f"  uv island margin: {cfg.uv_island_margin}")
    print(f"  ray distance:     {'auto' if cfg.ray_distance is None else cfg.ray_distance}")
    print(f"  cage extrusion:   {'auto' if cfg.cage_extrusion is None else cfg.cage_extrusion}")

    print("\n[1/10] Clear + Import")
    clear_scene()
    imported = import_glb(cfg.input_glb)

    print("\n[2/10] Join + Prepare source mesh")
    source = join_mesh_objects(imported)
    source.name = "HighPoly"
    apply_rotation_scale(source)
    source_tris = triangle_count(source)
    print(f"  source triangles: {source_tris}")

    print("\n[3/10] Voxel search for target triangle budget")
    best = find_best_voxel_candidate(source, cfg)
    lowpoly = best.obj

    ray_distance, cage_extrusion = resolve_bake_projection_distances(
        source,
        cfg,
        voxel_hint=best.voxel,
    )
    print(
        f"  bake projection: ray_distance={ray_distance:.6f}, cage_extrusion={cage_extrusion:.6f}"
    )

    print("\n[4/10] Remove floaters")
    if cfg.remove_floaters:
        remove_floating_components(
            lowpoly,
            min_triangles=cfg.floater_min_triangles,
            min_ratio=cfg.floater_min_ratio,
        )
    else:
        print("  floater removal: disabled")

    print("\n[5/10] UV unwrap lowpoly")
    unwrap_lowpoly_uv(
        lowpoly,
        method=cfg.uv_method,
        smart_angle_deg=cfg.uv_angle_deg,
        seam_angle_deg=cfg.uv_seam_angle_deg,
        island_margin=cfg.uv_island_margin,
    )

    print("\n[6/10] Apply normal smoothing")
    apply_normal_smoothing(lowpoly, cfg.smooth_angle_deg, cfg.weighted_normals)

    print("\n[7/10] Setup bake maps")
    setup_cycles(cfg.samples, cfg.force_cpu)
    output_dir = os.path.dirname(cfg.output_glb) or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    bake_images = {}

    diffuse = bpy.data.images.new("BakedDiffuse", cfg.bake_resolution, cfg.bake_resolution, alpha=False)
    diffuse.colorspace_settings.name = "sRGB"
    bake_images["DIFFUSE"] = diffuse

    if cfg.bake_normal:
        normal = bpy.data.images.new("BakedNormal", cfg.bake_resolution, cfg.bake_resolution, alpha=False)
        normal.colorspace_settings.name = "Non-Color"
        bake_images["NORMAL"] = normal

    if cfg.bake_roughness:
        rough = bpy.data.images.new("BakedRoughness", cfg.bake_resolution, cfg.bake_resolution, alpha=False)
        rough.colorspace_settings.name = "Non-Color"
        bake_images["ROUGHNESS"] = rough

    if cfg.bake_metallic:
        metallic = bpy.data.images.new("BakedMetallic", cfg.bake_resolution, cfg.bake_resolution, alpha=False)
        metallic.colorspace_settings.name = "Non-Color"
        bake_images["METALLIC"] = metallic

    if cfg.bake_ao:
        ao = bpy.data.images.new("BakedAO", cfg.bake_resolution, cfg.bake_resolution, alpha=False)
        ao.colorspace_settings.name = "Non-Color"
        bake_images["AO"] = ao

    if cfg.bake_emission:
        emission = bpy.data.images.new("BakedEmission", cfg.bake_resolution, cfg.bake_resolution, alpha=False)
        emission.colorspace_settings.name = "sRGB"
        bake_images["EMISSION"] = emission

    print("\n[8/10] Bake maps")
    bake_order = ["DIFFUSE", "NORMAL", "ROUGHNESS", "METALLIC", "AO", "EMISSION"]
    for bake_type in bake_order:
        image = bake_images.get(bake_type)
        if image is None:
            continue

        create_target_material_with_image(lowpoly, image, f"BakeMat_{bake_type}")
        bake_map(
            source,
            lowpoly,
            bake_type,
            image,
            ray_distance,
            cage_extrusion,
            cfg.bake_margin,
        )

        image_path = os.path.join(output_dir, f"baked_{bake_type.lower()}.png")
        image.filepath_raw = image_path
        image.file_format = "PNG"
        image.save()
        print(f"  saved {image_path}")

    # Final material assignment for GLB
    final_mat = bpy.data.materials.new(name="FinalMaterial")
    final_mat.use_nodes = True
    nodes = final_mat.node_tree.nodes
    links = final_mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    tex_diff = nodes.new("ShaderNodeTexImage")
    tex_diff.location = (-900, 0)
    tex_diff.image = bake_images["DIFFUSE"]

    base_color_socket = tex_diff.outputs["Color"]

    if "AO" in bake_images:
        tex_ao = nodes.new("ShaderNodeTexImage")
        tex_ao.location = (-900, -240)
        tex_ao.image = bake_images["AO"]
        tex_ao.image.colorspace_settings.name = "Non-Color"

        mix_ao = nodes.new("ShaderNodeMixRGB")
        mix_ao.location = (-620, -80)
        mix_ao.blend_type = "MULTIPLY"
        mix_ao.inputs["Factor"].default_value = 1.0
        links.new(tex_diff.outputs["Color"], mix_ao.inputs["Color1"])
        links.new(tex_ao.outputs["Color"], mix_ao.inputs["Color2"])
        base_color_socket = mix_ao.outputs["Color"]

    links.new(base_color_socket, bsdf.inputs["Base Color"])

    if "NORMAL" in bake_images:
        tex_n = nodes.new("ShaderNodeTexImage")
        tex_n.location = (-900, -480)
        tex_n.image = bake_images["NORMAL"]
        tex_n.image.colorspace_settings.name = "Non-Color"

        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.location = (-620, -480)
        links.new(tex_n.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    if "ROUGHNESS" in bake_images:
        tex_r = nodes.new("ShaderNodeTexImage")
        tex_r.location = (-900, -720)
        tex_r.image = bake_images["ROUGHNESS"]
        tex_r.image.colorspace_settings.name = "Non-Color"

        sep_r = nodes.new("ShaderNodeSeparateColor")
        sep_r.location = (-620, -720)
        if hasattr(sep_r, "mode"):
            sep_r.mode = "RGB"
        links.new(tex_r.outputs["Color"], sep_r.inputs["Color"])
        links.new(sep_r.outputs["Green"], bsdf.inputs["Roughness"])

    if "METALLIC" in bake_images:
        tex_m = nodes.new("ShaderNodeTexImage")
        tex_m.location = (-900, -960)
        tex_m.image = bake_images["METALLIC"]
        tex_m.image.colorspace_settings.name = "Non-Color"

        sep_m = nodes.new("ShaderNodeSeparateColor")
        sep_m.location = (-620, -960)
        if hasattr(sep_m, "mode"):
            sep_m.mode = "RGB"
        links.new(tex_m.outputs["Color"], sep_m.inputs["Color"])
        links.new(sep_m.outputs["Blue"], bsdf.inputs["Metallic"])

    if "EMISSION" in bake_images:
        tex_e = nodes.new("ShaderNodeTexImage")
        tex_e.location = (-900, -1200)
        tex_e.image = bake_images["EMISSION"]
        links.new(tex_e.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 1.0

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    lowpoly.data.materials.clear()
    lowpoly.data.materials.append(final_mat)

    print("\n[9/10] Transfer source custom normals")
    if cfg.transfer_source_normals:
        transfer_source_custom_normals(source, lowpoly)
    else:
        print("  source normal transfer: disabled")

    print("\n[10/10] Export")
    source.hide_set(True)
    export_glb(cfg.output_glb, lowpoly)

    final_tris = triangle_count(lowpoly)
    reduction = (1.0 - (final_tris / max(source_tris, 1))) * 100.0

    print("\n" + "=" * 72)
    print("DONE")
    print(f"  source triangles:  {source_tris}")
    print(f"  output triangles:  {final_tris}")
    print(f"  reduction:         {reduction:.2f}%")
    print(f"  chosen voxel size: {best.voxel:.6f}")
    print(f"  output file:       {cfg.output_glb}")
    print("=" * 72)


if __name__ == "__main__":
    main()
