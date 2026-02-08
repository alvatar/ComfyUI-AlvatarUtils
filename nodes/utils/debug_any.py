class DebugAny:
    """
    Debug node that displays type and info about any input in the UI.
    Also passes through the value unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*", {
                    "tooltip": "Any value to debug. Type, shape, and value will be shown in the node."
                }),
            }
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("output", "debug_text")
    OUTPUT_TOOLTIPS = (
        "Pass-through of input value unchanged.",
        "Debug information as text string.",
    )
    FUNCTION = "debug"
    CATEGORY = "Alvatar/Utils"
    DESCRIPTION = "Displays type, value, shape, and length of any input. Shows info in UI and passes value through unchanged."
    OUTPUT_NODE = True

    def debug(self, input):
        lines = []

        # Type info
        type_name = type(input).__name__
        module = type(input).__module__
        if module and module != 'builtins':
            type_name = f"{module}.{type_name}"
        lines.append(f"Type: {type_name}")

        # Shape (for tensors, arrays)
        if hasattr(input, 'shape'):
            lines.append(f"Shape: {input.shape}")

        # Length (for sequences)
        if hasattr(input, '__len__') and not isinstance(input, str):
            lines.append(f"Length: {len(input)}")

        # Dtype (for tensors)
        if hasattr(input, 'dtype'):
            lines.append(f"Dtype: {input.dtype}")

        # Device (for torch tensors)
        if hasattr(input, 'device'):
            lines.append(f"Device: {input.device}")

        # Value preview
        value_str = str(input)
        if len(value_str) > 500:
            value_str = value_str[:500] + "..."
        lines.append(f"Value: {value_str}")

        debug_text = "\n".join(lines)

        # Print to console as well
        print(f"[DebugAny] {debug_text}")

        # Return both the passthrough and the debug text
        return {"ui": {"text": [debug_text]}, "result": (input, debug_text)}
