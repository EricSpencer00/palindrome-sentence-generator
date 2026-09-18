"""Identify the actual current files without exposing workstation paths."""
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys


def environment():
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "system": platform.system(), "release": platform.release(),
            "architecture": platform.machine(), "optimization": sys.flags.optimize}


def snapshot(root, paths):
    root = Path(root).resolve()
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(paths)}
    git = None
    try:
        def command(*args):
            return subprocess.check_output(["git", *args], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        if Path(command("rev-parse", "--show-toplevel")).resolve() == root:
            git = {"head": command("rev-parse", "HEAD"),
                   "dirty": bool(command("status", "--porcelain", "--untracked-files=normal")),
                   "meaning": "Committed HEAD is context only; file hashes identify the current source."}
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"scope": "Current source snapshot; historical run source was not frozen.",
            "files_sha256": hashes,
            "snapshot_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "git": git, "environment": environment()}


def current_sources(root):
    root = Path(root)
    return sorted(path for directory in ("paper", "llm_palindrome", "experiments", "server")
                  for path in (root / directory).glob("*.py") if path.is_file())
