from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
BASE_PREFIX = Path(sys.base_prefix)
LIBRARY_BIN = BASE_PREFIX / "Library" / "bin"
REQUIRED_DLLS = [
    "ffi.dll",
    "ffi-7.dll",
    "ffi-8.dll",
    "libbz2.dll",
    "liblzma.dll",
    "sqlite3.dll",
]


def main() -> int:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--name",
        "api",
        "main.py",
    ]

    included = []
    for dll_name in REQUIRED_DLLS:
        dll_path = LIBRARY_BIN / dll_name
        if dll_path.exists():
            command.extend(["--add-binary", f"{dll_path};."])
            included.append(str(dll_path))

    print("Using Python:", sys.executable)
    print("Base prefix:", BASE_PREFIX)
    print("Including DLLs:")
    for item in included:
        print("-", item)

    completed = subprocess.run(command, cwd=PROJECT_DIR, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
