#!/usr/bin/env python3
"""Compatibility alias for the older mainline entry name.

Main branch keeps this wrapper only for compatibility.
Prefer `run_mainline_bundle.py` as the clean-main canonical entry.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().with_name("run_mainline_bundle.py")
    runpy.run_path(str(target), run_name="__main__")
