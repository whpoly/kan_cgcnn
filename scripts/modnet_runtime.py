"""Activate process-local Windows DLL paths when invoking an env's Python directly."""
import os
from pathlib import Path
import sys

_dll_handles = []


def configure_official_runtime():
    if os.name == "nt":
        root = Path(sys.prefix)
        paths = [root, root / "Library" / "mingw-w64" / "bin",
                 root / "Library" / "usr" / "bin", root / "Library" / "bin",
                 root / "Scripts"]
        existing = [str(path) for path in paths if path.is_dir()]
        os.environ["PATH"] = os.pathsep.join(existing + [os.environ.get("PATH", "")])
        for path in existing:
            if hasattr(os, "add_dll_directory"):
                _dll_handles.append(os.add_dll_directory(path))
