# eyetrack

Clean mainline for indoor pingpong-space eye-tracking analysis.

This branch (`main`) now focuses on **two output blocks only**:

1. `01_描述性分析_Descriptive`
2. `02_显著性分析_Significance`

The previous richer / messier structure is preserved on the **`raw` branch**.
If you need the old naming, exploratory folders, or full historical layout, switch to `raw`.

---

## Mainline scope

### 01_描述性分析_Descriptive
Primary reading path for descriptive outputs.

What is highlighted on `main`:
- `overall` / overall descriptive organization
- `Experience` grouping

What is intentionally de-emphasized on `main`:
- older cross-cutting folder variants
- extra historical naming around `analysis2`, `research_bundle`, and multi-stage bundle wording
- non-primary grouped views not needed for the current paper narrative

### 02_显著性分析_Significance
Primary reading path for inferential outputs.

Main contents:
- allocation LMM outputs
- explanatory LMM visuals
- two-part models (when scene features are available)

### Appendix / legacy / raw-preserved content
Still kept, but no longer presented as the main story:
- diagnostics
- raw AOI batch outputs
- Colab notes / implementation notes
- compatibility wrappers with old names

---

## Branches

### `main`
For the cleaner paper-facing workflow.

### `raw`
Snapshot of the repository before this cleanup.
Use it when you need:
- the previous README structure
- the older folder naming logic
- fuller historical context
- non-mainline exploratory material without reorganization

---

## Recommended commands

### Local / VPS mainline run

```bash
cd /root/.openclaw/workspace/eyetrack
source .venv/bin/activate
python scripts/run_analysis2.py \
  --group_manifest /path/to/group_manifest.csv \
  --scenes_root /path/to/scenes_root
```

### Colab one-command run

```bash
cd /content/eyetrack
python scripts/run_colab_one_command.py \
  --scenes_root_orig /content/drive/MyDrive/映射 \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv
```

---

## Current output structure

```text
研究输出_YYYYMMDD_HHMMSS/
├─ 01_描述性分析_Descriptive/
│  ├─ organized_outputs/
│  ├─ grouped_overall/
│  └─ grouped_experience/
├─ 02_显著性分析_Significance/
│  ├─ allocation_lmm/
│  ├─ allocation_lmm_visuals/
│  └─ two_part_models/
└─ 98_附录_Appendix/
   ├─ diagnostics/
   ├─ raw_batch_outputs/
   ├─ notes/
   └─ colab_notes/
```

---

## What is mainline vs downgraded

### Mainline
- `01_描述性分析_Descriptive/organized_outputs`
- `01_描述性分析_Descriptive/grouped_overall`
- `01_描述性分析_Descriptive/grouped_experience`
- `02_显著性分析_Significance/allocation_lmm`
- `02_显著性分析_Significance/allocation_lmm_visuals`
- `02_显著性分析_Significance/two_part_models`

### Downgraded but kept
- old “research bundle” wording
- old “analysis2” naming as the user-facing story
- raw batch AOI exports
- diagnostics and implementation notes
- compatibility wrappers such as older alias scripts

Nothing useful was intentionally deleted; non-mainline material is either preserved on `raw` or kept under appendix/compatibility paths.

---

## Minimal setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Main entry scripts

- `scripts/run_analysis2.py` — current mainline orchestrator
- `scripts/run_colab_one_command.py` — preferred Colab entry
- `scripts/run_aoi_metrics.py` — AOI metrics for single-run use
- `scripts/run_minimal_aoi_bundle.py` — minimal four-input AOI bundle

Compatibility aliases remain in the repo, but they are not the recommended top-level story on `main`.

---

## Legacy notes

Some older docs and helper scripts still contain terms such as:
- `research_bundle`
- `analysis2`
- `03_TwoPart模型`
- `00_AOI原始批处理`

Those names are retained only for compatibility or historical continuity. The `main` branch narrative should now be read through the two-block structure above.
