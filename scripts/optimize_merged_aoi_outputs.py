#!/usr/bin/env python3
"""Compatibility alias for the older merged-output optimizer entry name.

Prefer `optimize_merged_batch_outputs.py` on the clean main branch.
"""

import runpy
from pathlib import Path


if __name__ == '__main__':
    target = Path(__file__).resolve().with_name('optimize_merged_batch_outputs_impl.py')
    runpy.run_path(str(target), run_name='__main__')
