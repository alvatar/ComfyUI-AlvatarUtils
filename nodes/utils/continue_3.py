"""
Continue3 - Synchronization barrier with 3 passthrough channels.

Waits for all connected inputs to complete before passing them through.
Use this to synchronize parallel branches in your workflow.

Why This Node Exists:
═════════════════════
ComfyUI executes nodes as soon as their inputs are ready. This is efficient
but can cause problems:

1. VRAM Competition: Two heavy nodes (Trellis2 + Upscale) might run simultaneously,
   causing OOM errors. Continue3 forces them to wait for each other.

2. Execution Order: You want A→B→C but there's no data dependency. Connect A's
   output to Continue3, then Continue3's output (or any connected input) to B.

3. Parallel Sync: Three branches process independently, but you need them ALL
   done before the next stage. Connect all three to Continue3.

This is the same concept as Trellis2's "Continue" node, but with 3 channels
instead of 2 for more complex workflows.
"""


class Continue3:
    """
    Synchronization barrier with 3 passthrough channels.

    How It Works:
    ═════════════
    ComfyUI won't execute this node until ALL connected inputs are ready.
    Once executed, it simply passes each input to its corresponding output.
    This creates a "checkpoint" where parallel branches must sync up.

    Why 3 Channels?
    ═══════════════
    Many 3D workflows have 3 parallel outputs:
    - Mesh generation (Trellis2, UltraShape)
    - Texture generation (Chord, Hunyuan3D texturing)
    - Reference image processing (Upscale, etc.)

    All three often need to complete before saving/exporting.

    Typical Use Cases:
    ══════════════════
    1. VRAM Management:
       [Heavy GPU Node A] ─→ input1    output1 ─→ [Heavy GPU Node B]
       Forces A to finish and release VRAM before B starts.

    2. Multi-Branch Sync:
       [Trellis2] ─────→ input1    output1 ─→ [SaveMesh]
       [Upscale] ──────→ input2    output2 ─→ [SaveImage]
       [Chord] ────────→ input3    output3 ─→ [Apply Textures]
       All three must complete before ANY downstream node runs.

    3. Execution Ordering:
       [Generate] ─→ input1 (used)      output1 ─→ [Process]
                     input2 (unused)    output2 (ignored)
       Even with one connection, creates a dependency checkpoint.

    Note: Unconnected inputs pass through as None. Connect what you need.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "input1": ("*", {
                    "tooltip": (
                        "First passthrough channel. Accepts any data type. "
                        "Passed to output1 after ALL connected inputs complete. "
                        "Leave unconnected if you don't need this channel."
                    )
                }),
                "input2": ("*", {
                    "tooltip": (
                        "Second passthrough channel. Accepts any data type. "
                        "Passed to output2 after ALL connected inputs complete. "
                        "Leave unconnected if you don't need this channel."
                    )
                }),
                "input3": ("*", {
                    "tooltip": (
                        "Third passthrough channel. Accepts any data type. "
                        "Passed to output3 after ALL connected inputs complete. "
                        "Leave unconnected if you don't need this channel."
                    )
                }),
            }
        }

    RETURN_TYPES = ("*", "*", "*")
    RETURN_NAMES = ("output1", "output2", "output3")
    OUTPUT_TOOLTIPS = (
        "Passthrough of input1. Available after all connected inputs complete.",
        "Passthrough of input2. Available after all connected inputs complete.",
        "Passthrough of input3. Available after all connected inputs complete.",
    )
    FUNCTION = "execute"
    CATEGORY = "Alvatar/Utils"
    DESCRIPTION = (
        "Synchronization barrier: waits for all connected inputs to complete, "
        "then passes them through. Use for VRAM management (sequential GPU ops) "
        "or to sync parallel branches before the next stage."
    )

    def execute(self, input1=None, input2=None, input3=None):
        """
        Pass through all inputs after they complete.
        ComfyUI ensures all connected inputs are ready before this executes.
        """
        return (input1, input2, input3)


# Keep backward compatibility alias
ConditionalExecution = Continue3
