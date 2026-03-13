#!/usr/bin/env python3
"""Compatibility alias for the older scene-feature mixed-model entry name.

Prefer `model_scene_feature_mixed.py` on the clean main branch.
"""

import runpy
from pathlib import Path


if __name__ == '__main__':
    target = Path(__file__).resolve().with_name('model_scene_feature_mixed_impl.py')
    runpy.run_path(str(target), run_name='__main__')
