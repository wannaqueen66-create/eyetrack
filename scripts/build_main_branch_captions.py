#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CAPTIONS = [
    {
        "label": "Fig01_experience_stability_overview",
        "title": "Stability overview of the Experience-centered LMM packet",
        "caption_en": "Overview of model stability across the three Experience-centered LMM families. This figure is suitable as an entry figure before discussing individual outcomes, because it quickly shows which outcome-family combinations are stable enough for main-text interpretation.",
        "caption_zh": "Experience 主线 LMM 三套模型家族的稳定性总览。适合作为进入各个 outcome 之前的总览图，先交代哪些 outcome × family 组合足够稳定，可以进入正文解释。",
    },
    {
        "label": "Fig02_experience_model_fit_overview",
        "title": "Model-fit overview of the Experience inferential packet",
        "caption_en": "Model-fit overview across the Experience inferential packet. This figure helps separate outcomes with interpretable fit from those that should be treated more cautiously or moved to supplementary discussion.",
        "caption_zh": "Experience 主显著性链条的模型拟合总览。用于区分哪些 outcome 的模型拟合足够可解释，哪些结果需要更谨慎处理或转入补充材料。",
    },
    {
        "label": "Fig03_grouped_experience_share_pct",
        "title": "Experience-group descriptive pattern for share_pct",
        "caption_en": "Grouped descriptive pattern for share_pct by Experience. This figure is primarily descriptive and should be used to introduce the intuitive direction and magnitude of group differences before turning to LMM-based inferential results.",
        "caption_zh": "按 Experience 分组的 share_pct 描述性模式图。主要用于先交代组间差异的直观方向和大小，随后再进入 LMM 推断结果。",
    },
    {
        "label": "Fig04_grouped_experience_fc_share",
        "title": "Experience-group descriptive pattern for FC_share",
        "caption_en": "Grouped descriptive pattern for FC_share by Experience. This figure complements TFD-based allocation share and helps show whether count-based allocation follows a convergent descriptive pattern.",
        "caption_zh": "按 Experience 分组的 FC_share 描述性模式图。可作为 TFD 分配占比的补充，用来判断基于注视次数的分配模式是否与之收敛。",
    },
    {
        "label": "Fig05_grouped_experience_ttff_y",
        "title": "Experience-group descriptive pattern for ttff_y",
        "caption_en": "Grouped descriptive pattern for ttff_y by Experience. This figure is useful when the manuscript wants to contrast attention allocation with attention-entry latency.",
        "caption_zh": "按 Experience 分组的 ttff_y 描述性模式图。适用于正文需要把注意分配与首次进入时延并列讨论的情形。",
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description='Build draft figure captions for eyetrack main-branch figure packs')
    ap.add_argument('--out-dir', type=Path, default=Path('docs'))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    md_lines = ['# Main-branch Figure Caption Drafts', '']
    for item in DEFAULT_CAPTIONS:
        md_lines.append(f"## {item['label']} — {item['title']}")
        md_lines.append('')
        md_lines.append('**EN**')
        md_lines.append(item['caption_en'])
        md_lines.append('')
        md_lines.append('**ZH**')
        md_lines.append(item['caption_zh'])
        md_lines.append('')

    md_path = args.out_dir / 'MAIN_BRANCH_FIGURE_CAPTIONS.md'
    json_path = args.out_dir / 'main_branch_figure_captions.json'
    md_path.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
    json_path.write_text(json.dumps(DEFAULT_CAPTIONS, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'generated': [str(md_path), str(json_path)]}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
