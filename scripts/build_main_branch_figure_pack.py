#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def candidate_specs() -> list[tuple[str, list[str], str]]:
    return [
        (
            "Fig01_experience_stability_overview",
            [
                "02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/evidence_stability_overview_Experience.png",
                "02_显著性分析_Significance/allocation_lmm/groupvar_Experience/01_main_effects/png/evidence_stability_overview_Experience.png",
            ],
            "Experience LMM stability overview for the three model families.",
        ),
        (
            "Fig02_experience_model_fit_overview",
            [
                "02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/evidence_model_fit_overview_Experience.png",
                "02_显著性分析_Significance/allocation_lmm/groupvar_Experience/01_main_effects/png/evidence_model_fit_overview_Experience.png",
            ],
            "Model-fit overview across the main Experience inferential packet.",
        ),
        (
            "Fig03_grouped_experience_share_pct",
            ["01_描述性分析_Descriptive/grouped_experience/png/plot_Experience_share_pct.png"],
            "Descriptive grouped pattern for share_pct by Experience.",
        ),
        (
            "Fig04_grouped_experience_fc_share",
            ["01_描述性分析_Descriptive/grouped_experience/png/plot_Experience_FC_share.png"],
            "Descriptive grouped pattern for FC_share by Experience.",
        ),
        (
            "Fig05_grouped_experience_ttff_y",
            ["01_描述性分析_Descriptive/grouped_experience/png/plot_Experience_ttff_y.png"],
            "Descriptive grouped pattern for ttff_y by Experience.",
        ),
    ]


TRACKS = [
    ("all_sample", "00_全样本_AllSample"),
    ("after_qc", "01_QC后_AfterQC"),
]


def build_track(results_root: Path, track_slug: str, track_dirname: str, out_dir: Path) -> dict:
    src_root = results_root / track_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    missing = []
    for label, rel_options, purpose in candidate_specs():
        src = None
        chosen_rel = None
        for rel in rel_options:
            cand = src_root / rel
            if cand.exists():
                src = cand
                chosen_rel = rel
                break
        if src is None or chosen_rel is None:
            missing.append(rel_options[0])
            continue
        dst = out_dir / f"{label}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        selected.append(
            {
                "label": label,
                "file": dst.name,
                "source": str(src.relative_to(results_root)),
                "purpose": purpose,
                "matched_from": chosen_rel,
            }
        )

    lines = [
        f"# Main-branch figure pack — {track_slug}",
        "",
        "Suggested manuscript-facing figures copied from the current eyetrack clean-main result bundle.",
        "",
        "| Label | File | Source | Use in manuscript |",
        "|---|---|---|---|",
    ]
    for row in selected:
        lines.append(f"| {row['label']} | `{row['file']}` | `{row['source']}` | {row['purpose']} |")
    (out_dir / "FIGURE_PACK_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "track": track_slug,
        "track_dirname": track_dirname,
        "selected": selected,
        "missing": missing,
        "n_selected": len(selected),
    }
    (out_dir / "figure_pack_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a light main-branch figure pack for eyetrack result bundles")
    ap.add_argument("--results-root", type=Path, required=True, help="Path to a generated 研究输出_时间戳 folder")
    ap.add_argument("--out-dir", type=Path, default=Path("figure_pack_main_branch"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for slug, dirname in TRACKS:
        summary[slug] = build_track(args.results_root, slug, dirname, args.out_dir / slug)
    (args.out_dir / "figure_pack_manifest_all.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "tracks": list(summary.keys())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
