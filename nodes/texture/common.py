"""
Shared utilities for Blender-based texture baking nodes.
"""

import os
import subprocess

# Plugin info
__version__ = "1.0.0"
__author__ = "alvatar"


def log(msg):
    """Plugin logger with prefix"""
    print(f"[Alvatar/Texture] {msg}")


def check_blender():
    """Check if Blender is available and return (path, version_string)"""
    blender_paths = [
        "/usr/bin/blender",
        "/usr/sbin/blender",
        "/usr/local/bin/blender",
        "/opt/blender/blender",
    ]

    for path in blender_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    version_line = result.stdout.split('\n')[0]
                    return path, version_line
            except:
                pass

    # Try PATH
    try:
        result = subprocess.run(
            ["blender", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            return "blender", version_line
    except:
        pass

    return None, None


# Check Blender on module load
BLENDER_PATH, BLENDER_VERSION = check_blender()
