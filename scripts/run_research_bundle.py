#!/usr/bin/env python3
"""Preferred alias entry for the canonical eyetrack research bundle.

This wrapper preserves compatibility while exposing a clearer name than the
legacy `run_analysis2.py` entry.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().with_name("run_analysis2.py")
    runpy.run_path(str(target), run_name="__main__")
