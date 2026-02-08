import os
import folder_paths


class ResolvePath:
    """
    Resolves a relative path to absolute by checking output, input, and temp directories.
    Fixes compatibility between nodes that save to output/ and nodes that expect input/.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "",
                    "tooltip": "Relative or absolute file path. Will search output/, input/, temp/ directories."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("absolute_path",)
    OUTPUT_TOOLTIPS = ("Resolved absolute path to the file, or original path if not found.",)
    FUNCTION = "resolve"
    CATEGORY = "Alvatar/Utils"
    DESCRIPTION = "Converts relative paths to absolute by searching output/, input/, temp/ directories. Fixes compatibility between nodes that save vs load files."

    def resolve(self, path):
        # Already absolute and exists
        if os.path.isabs(path) and os.path.exists(path):
            return (path,)

        # Try output directory first (most common for generated files)
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, path)
        if os.path.exists(output_path):
            print(f"[ResolvePath] Found in output: {output_path}")
            return (output_path,)

        # Try input directory
        input_dir = folder_paths.get_input_directory()
        input_path = os.path.join(input_dir, path)
        if os.path.exists(input_path):
            print(f"[ResolvePath] Found in input: {input_path}")
            return (input_path,)

        # Try temp directory
        temp_dir = folder_paths.get_temp_directory()
        temp_path = os.path.join(temp_dir, path)
        if os.path.exists(temp_path):
            print(f"[ResolvePath] Found in temp: {temp_path}")
            return (temp_path,)

        # Return original if nothing found (let downstream node handle error)
        print(f"[ResolvePath] WARNING: Could not resolve path: {path}")
        return (path,)
