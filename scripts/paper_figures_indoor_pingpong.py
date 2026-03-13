#!/usr/bin/env python3
"""Compatibility alias for the older manuscript-figure entry name.

Prefer `build_manuscript_figures.py` on the clean main branch.
"""

import runpy
from pathlib import Path


if __name__ == '__main__':
    target = Path(__file__).resolve().with_name('build_manuscript_figures_impl.py')
    runpy.run_path(str(target), run_name='__main__')
