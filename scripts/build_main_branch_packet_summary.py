#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEXT = """# Main-branch Packet Summary (eyetrack)

## Primary inferential packet
- Experience LMM index: `02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/model_family_index.csv`
- Experience packet summary: `02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/three_model_packet_summary.csv`

## Primary outcomes
- share_pct
- share_logit
- FC_share
- fc_share_logit
- FC_rate
- tfd_y
- ttff_y
- fc_y

## Supplement / exploratory outcomes
- ffd_y
- mfd_y
- rff_y
- MPD

## Reading rule
1. model fit
2. fixed effects
3. trend coding
4. contrasts
5. evidence png
"""


def main() -> int:
    ap = argparse.ArgumentParser(description='Generate a lightweight packet summary for eyetrack main')
    ap.add_argument('--out-dir', type=Path, default=Path('docs'))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / 'MAIN_BRANCH_PACKET_SUMMARY.md'
    out.write_text(TEXT, encoding='utf-8')
    print(json.dumps({'generated': str(out)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
