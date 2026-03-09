#!/usr/bin/env python3
"""Legacy alias for the older research-bundle naming.

Main branch keeps this wrapper only for compatibility.
Prefer `run_analysis2.py` or the clearer `run_colab_one_command.py` path documented in README.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().with_name("run_analysis2.py")
    runpy.run_path(str(target), run_name="__main__")
