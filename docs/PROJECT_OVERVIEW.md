# Project Overview

This document explains what the current `main` branch of the eye-tracking repository is trying to do, and how raw eye-tracking exports are turned into paper-facing result bundles.

---

## One-sentence summary
The current `main` branch is a **clean paper-facing analysis branch** for indoor pingpong-space eye-tracking data.

Its default goal is not to expose every historical exploratory output, but to produce a cleaner reading surface built around:

1. **full sample**
2. **after-QC re-run**

And inside each track:
- **descriptive analysis**
- **significance analysis**

---

## What problem this repo solves
The repo takes eye-tracking exports plus AOI definitions and turns them into a stable analysis bundle that supports:
- AOI metric extraction
- grouped descriptive summaries
- LMM-based inferential analysis
- optional two-part modeling when scene features are available
- cleaner result folders for manuscript reading and reporting

---

## Current main data flow

### 1) Input layer
Typical inputs are:
- `group_manifest.csv`
- scene-root folders with per-scene eye-tracking CSV files
- AOI JSON definitions
- optional scene-feature table

These are the raw ingredients for the mainline run.

### 2) AOI metric extraction
Primary scripts:
- `scripts/batch_aoi_metrics.py`
- `scripts/run_aoi_metrics.py`

This stage converts raw scene-level gaze/fixation data into AOI-level metrics such as:
- FC
- FFD
- MFD
- MPD
- RFF
- TFD
- TTFF
- normalized companions such as `share_pct`, `FC_share`, `FC_rate`

### 3) Mainline orchestration
Primary script:
- `scripts/run_mainline_bundle.py`

Compatibility alias:
- `scripts/run_analysis2.py`
- `scripts/run_research_bundle.py`

This is the canonical mainline orchestrator for the current `main` branch.
It creates two explicit tracks:
- `00_全样本_AllSample`
- `01_QC后_AfterQC`

Important:
- the QC track is **re-run from a filtered manifest**
- it is not just a figure-level hide/filter
- the exclusion list lives in `configs/excluded_participants_qc.csv`

### 4) Descriptive outputs
Main folders inside each result track:
- `01_描述性分析_Descriptive/organized_outputs`
- `01_描述性分析_Descriptive/grouped_overall`
- `01_描述性分析_Descriptive/grouped_experience`

Purpose:
- organize AOI outputs into a more readable structure
- provide grouped summaries and PNGs
- keep exact values inspectable via companion CSV data files

### 5) Significance outputs
Main folders inside each result track:
- `02_显著性分析_Significance/allocation_lmm`
- `02_显著性分析_Significance/allocation_lmm_visuals`
- `02_显著性分析_Significance/two_part_models`

Purpose:
- build the inferential mainline around allocation LMMs
- provide explanatory visuals as companion material
- extend to two-part models when scene-feature questions matter

### 6) Appendix / preserved layers
Also kept inside each result track:
- diagnostics
- raw batch outputs
- notes
- colab notes

These are preserved for checking, reproducibility, and deeper review, but they are not the main story of the current branch.

---

## Mainline interpretation policy

### Descriptive mainline
Descriptive outputs are mainly used for:
- overall level and pattern
- grouped context
- intuitive shape reading before exact numeric confirmation

### Inferential mainline
Inferential claims should mainly come from:
- `allocation_lmm/groupvar_Experience`

In other words:
- `overall` is useful context
- `Experience` is the current main inferential reading path

---

## Outcome priority on main

### Primary headline outcomes
These are currently the preferred paper-facing outcomes:
- `share_pct`
- `share_logit`
- `FC_share`
- `fc_share_logit`
- `FC_rate`
- `tfd_y`
- `ttff_y`
- `fc_y`

### Secondary / exploratory outcomes
These are better treated as supplement or mechanism-support outputs unless the study specifically targets them:
- `ffd_y`
- `mfd_y`
- `rff_y`
- `MPD`

---

## Branch policy

### `main`
Use this branch when you want:
- a cleaner paper-facing workflow
- explicit full-sample vs after-QC tracks
- a more guided reading surface

### `raw`
Use this branch when you want:
- older naming and folder logic
- broader historical exploratory outputs
- legacy structure without current cleanup/reframing

---

## Recommended entry files for new readers
If someone opens this repo for the first time, the recommended reading order is:
1. `README.md`
2. `README_zh.md`
3. `RESULTS_MAP.md`
4. `docs/RESULTS_READING_GUIDE.md`
5. `docs/RESULTS_READING_GUIDE.zh.md`
6. `docs/SIGNIFICANCE_MAINLINE.md`

---

## Minimal operational summary
If you only remember three things, remember these:
1. `run_mainline_bundle.py` is the current canonical mainline entry
2. `main` is centered on full-sample + after-QC result tracks
3. inferential claims should be read primarily from the `Experience` LMM mainline, not from `overall` descriptive outputs alone
