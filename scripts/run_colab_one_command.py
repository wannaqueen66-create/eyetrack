#!/usr/bin/env python3
"""Preferred Colab one-command entry for eyetrack.

This is the clean-main alias. The underlying implementation remains in
`run_colab_mainline_bundle.py`.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().with_name("run_colab_mainline_bundle.py")
    runpy.run_path(str(target), run_name="__main__")
