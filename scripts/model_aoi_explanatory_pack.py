#!/usr/bin/env python3
"""Canonical explanatory AOI modeling entry for clean main."""

import runpy
from pathlib import Path


if __name__ == '__main__':
    target = Path(__file__).resolve().with_name('model_aoi_explanatory_pack_impl.py')
    runpy.run_path(str(target), run_name='__main__')
