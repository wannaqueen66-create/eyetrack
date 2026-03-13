#!/usr/bin/env python3
"""Compatibility alias for the older one-scene AOI bundle entry name.

Prefer `run_one_scene_bundle.py` on the clean main branch.
"""

import runpy
from pathlib import Path


if __name__ == '__main__':
    target = Path(__file__).resolve().with_name('run_one_scene_bundle_impl.py')
    runpy.run_path(str(target), run_name='__main__')
