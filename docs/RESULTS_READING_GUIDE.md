# Results Reading Guide (eyetrack main)

This is the fastest reading entry for collaborators, supervisors, or anyone who has received a `研究输出_时间戳/` folder from the current eye-tracking `main` branch and wants to know where to start.

---

## One-sentence summary
The current `main` branch organizes eye-tracking results into two explicit tracks:

1. **full sample**
2. **after-QC re-run**

Inside each track, only two major reading blocks are emphasized:
- **descriptive analysis**
- **significance analysis**

For formal reporting, prefer the **after-QC** track.

---

## Step 1 — Start with the global track guide
When you open a result bundle, read these first:
1. `研究输出_时间戳/README_研究输出说明.txt`
2. `研究输出_时间戳/01_QC后_AfterQC/README_研究输出说明.txt`

Why:
- they explain what the two tracks mean
- they clarify that the QC track is re-run from a filtered manifest
- they tell you how the result tree is split into descriptive vs significance sections

---

## Step 2 — Read descriptive outputs first

### 2.1 Full-sample context
Start with:
- `研究输出_时间戳/00_全样本_AllSample/01_描述性分析_Descriptive/grouped_overall/png/`

Then confirm with:
- `.../grouped_overall/data/`
- `.../grouped_overall/tables/`

What this gives you:
- overall level and pattern
- whether the distributions are visually stable or noisy
- a quick sense of which outcomes deserve inferential attention

### 2.2 Experience-group descriptive picture
For the current mainline, this is usually more important than overall descriptives.

Start with:
- `研究输出_时间戳/01_QC后_AfterQC/01_描述性分析_Descriptive/grouped_experience/png/`

Then confirm with:
- `.../grouped_experience/data/`
- `.../grouped_experience/tables/`

What this gives you:
- the intuitive gap between Experience groups
- which outcomes may be promising for the inferential mainline

Simple rule:
- **PNG first for pattern**
- **CSV/data second for exact values**

---

## Step 3 — Read the inferential mainline

### 3.1 Start with the Experience LMM index
Highest priority:
- `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/model_family_index.csv`
- `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/three_model_packet_summary.csv`

These tell you:
- how the LMM results are organized into three model families
- which outcome × family combinations are stable enough to read as main-result candidates

### 3.2 Then enter the three model families in order
Read in this sequence:
1. `01_main_effects`
2. `02_two_way_interactions`
3. `03_three_way_interaction`

This keeps the statistical story stable and prevents jumping too early into the most complex interactions.

---

## Step 4 — For each outcome, use a fixed reading packet
Within each model family, read files in this order:
1. `model_stability_summary.csv`
2. `model_fit_<outcome>.csv`
3. `fixef_<outcome>.csv`
4. `wwr_trend_tests_<outcome>.csv` / `wwr_trend_coding_<outcome>.csv`
5. `contrasts_<outcome>.csv` (especially for interaction families)
6. `evidence_*.png`

Why this order works:
- first check whether the model is usable
- then read the fixed effects
- then inspect the trend coding and contrasts
- finally use the evidence PNGs for communication and explanation

---

## Step 5 — Prioritize the main outcomes first
Recommended primary outcome order on the current main branch:
1. `share_pct`
2. `share_logit`
3. `FC_share`
4. `fc_share_logit`
5. `FC_rate`
6. `tfd_y`
7. `ttff_y`
8. `fc_y`

These are the most suitable headline outcomes for the present paper-facing mainline.

---

## Step 6 — Treat some outcomes as supplement/mechanism-first
Usually read these later unless they are central to the exact study question:
- `ffd_y`
- `mfd_y`
- `rff_y`
- `MPD`

They are valuable, but they are better framed as:
- exploratory
- mechanism-support
- supplement-facing

---

## Step 7 — Use explanatory visuals correctly
Go to:
- `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm_visuals/`

Use this folder for:
- explanatory companion figures
- manuscript-friendly communication graphics
- visual interpretation support

Do **not** use it as a replacement for the core statistical tables.

---

## Step 8 — When to use two-part models
Go to:
- `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/two_part_models/`

Use these when:
- scene-feature questions are central
- zero-inflation / two-part structure is scientifically important
- you need an extension beyond the allocation LMM mainline

If scene-feature questions are secondary, these outputs are often better kept in supplement.

---

## Step 9 — How to split roles between overall and Experience
A very important rule on the current `main` branch:

- `overall` = descriptive context
- `Experience` = primary inferential path

In other words:
- do not build the main statistical claim from `grouped_overall` alone
- use `overall` for level/shape/context
- use `Experience` LMM outputs for the main inferential story

---

## Shortest practical route for a formal update
If you need the shortest reporting route, read in this order:
1. `研究输出_时间戳/README_研究输出说明.txt`
2. `研究输出_时间戳/01_QC后_AfterQC/01_描述性分析_Descriptive/grouped_experience/png/`
3. `.../allocation_lmm/groupvar_Experience/tables/model_family_index.csv`
4. `.../allocation_lmm/groupvar_Experience/tables/three_model_packet_summary.csv`
5. primary outcomes in the three model families
6. only then explanatory PNGs and optional two-part outputs

---

## Minimal summary
If you only remember four things, remember these:
1. formal reporting should start from the after-QC track
2. read descriptive PNGs before exact tables
3. read `Experience` LMM tables before explanatory visuals
4. treat exploratory outcomes as support unless the study question explicitly centers on them
