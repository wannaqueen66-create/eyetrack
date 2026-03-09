# eyetrack

Python toolkit for indoor pingpong-space eye-tracking analysis, aligned to the target journal *Building and Environment*.  
面向室内乒乓球空间场景的眼动分析工具（Python），并按目标期刊 *Building and Environment* 的投稿规范对齐。

---

## Table of Contents / 目录

- [1. Overview / 项目概览](#1-overview--项目概览)
- [2. Features / 核心功能](#2-features--核心功能)
- [3. Project Structure / 项目结构](#3-project-structure--项目结构)
- [4. Quick Start / 快速开始](#4-quick-start--快速开始)
- [5. End-to-End Workflow / 全流程操作](#5-end-to-end-workflow--全流程操作)
  - [Step A. Setup Environment / 环境准备](#step-a-setup-environment--环境准备)
  - [Step B. Run Basic Analysis / 运行基础分析](#step-b-run-basic-analysis--运行基础分析)
  - [Step C. Draw AOIs in Browser / 网页框选AOI](#step-c-draw-aois-in-browser--网页框选aoi)
  - [Step D. Compute AOI Metrics / 计算AOI指标](#step-d-compute-aoi-metrics--计算aoi指标)
  - [Step E. Read Results / 解读结果](#step-e-read-results--解读结果)
- [6. AOI Web Tool v3 Guide / AOI网页工具v3说明](#6-aoi-web-tool-v3-guide--aoi网页工具v3说明)
- [7. Input Data Requirements / 输入数据要求](#7-input-data-requirements--输入数据要求)
- [8. FAQ / 常见问题](#8-faq--常见问题)
- [9. Paper-Oriented Next Steps / 论文向下一步](#9-paper-oriented-next-steps--论文向下一步)
- [10. Optimized Output Workflow / 优化后的输出流程](#10-optimized-output-workflow--优化后的输出流程)
- [11. Building and Environment Figure Style Parameters / B&E图形规范参数表](#11-building-and-environment-figure-style-parameters--be图形规范参数表)

---

## 1. Overview / 项目概览

**EN**: This repository helps beginners go from exported eye-tracking CSV data to clean metrics and publication-ready visualizations for **indoor pingpong-space studies**, without depending on vendor post-processing software. The writing/output style is aligned to **Building and Environment** as the target journal.

**中文**：本仓库帮助你从导出的眼动 CSV 数据出发，完成面向**室内乒乓球空间研究**的清洗、可视化和 AOI 指标统计，全流程不依赖厂商后处理软件；方法与输出风格按 **Building and Environment** 目标期刊对齐。

---

## 2. Features / 核心功能

**EN**

- Data cleaning for eye-tracking CSV
- Heatmap and scanpath visualization
- AOI polygon labeling in browser
- Multiple separated polygons under one AOI class
- Metrics output by polygon and by class:
  - TFD (Total Fixation Duration)
  - TFF (Time to First Fixation)
  - FC (Fixation Count)
  - FFD (First Fixation Duration)
  - MFD (Mean Fixation Duration)
  - RFF (Re-fixation Frequency)
  - MPD (Mean Pupil Diameter)

**中文**

- 眼动 CSV 数据清洗
- 热图与 scanpath 可视化
- 浏览器内多边形 AOI 标注
- 同一 AOI 类支持多个分离区域
- 指标输出支持“子区域级”和“类别级”：
  - TFD（Total Fixation Duration，总注视时长）
  - TFF（Time to First Fixation，首次注视时间）
  - FC（Fixation Count，注视次数）
  - FFD（First Fixation Duration，首次注视时长）
  - MFD（Mean Fixation Duration，平均注视时长）
  - RFF（Re-fixation Frequency，重注视频率）
  - MPD（Mean Pupil Diameter，平均瞳孔直径）

---

## 3. Project Structure / 项目结构

```text
eyetrack/
├─ data/
│  ├─ raw/                # raw input files / 原始数据
│  └─ processed/          # processed data / 清洗后数据
├─ figures/               # exported figures / 导出图像
├─ outputs/               # script outputs / 脚本输出
├─ scripts/
│  ├─ run_pipeline.py     # basic pipeline / 基础分析
│  └─ run_aoi_metrics.py  # AOI metrics from aoi.json
├─ src/
│  ├─ pipeline.py
│  └─ aoi_metrics.py
└─ (AOI web app moved to separate repo)
   └─ https://github.com/wannaqueen66-create/eyetrack-aoi
```

---

## 4. Quick Start / 快速开始

```bash
cd /home/wannaqueen66/.openclaw/workspace/eyetrack
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_pipeline.py \
  --input /home/wannaqueen66/raw_User1_260128181841_0208103256.csv \
  --outdir outputs
```

Then use the standalone AOI web app (integrated from `eyetrack-aoi`) / 然后使用独立 AOI 网页工具（已对接 `eyetrack-aoi`）：

- Repo / 仓库: `https://github.com/wannaqueen66-create/eyetrack-aoi`
- Recommended / 推荐: deploy by Cloudflare Pages and open the `*.pages.dev` URL
- Optional local run / 可选本地运行:

```bash
git clone https://github.com/wannaqueen66-create/eyetrack-aoi.git
cd eyetrack-aoi/public
python3 -m http.server 8080
# open / 打开: http://<VPS-IP>:8080
```

---


## Colab Deployment / Colab 部署（Detailed / 详细流程）

Colab is the recommended way to **run quickly without setting up Python on your local machine/VPS**.
Colab 推荐用于**无需本地/VPS 配环境**的快速试跑。

### 0) Open notebook / 打开 Notebook

- Standard (mixed EN/中文):
  https://colab.research.google.com/github/wannaqueen66-create/eyetrack/blob/main/notebooks/eyetrack_colab_quickstart.ipynb
- Chinese step-by-step version:
  https://colab.research.google.com/github/wannaqueen66-create/eyetrack/blob/main/notebooks/eyetrack_colab_quickstart_zh.ipynb

> Full Chinese guide (includes batch + grouped diff + background overlay): `docs/COLAB_ZH.md`
>
> Copy-by-cell markdown (no notebook, just paste cell by cell): `docs/COLAB_CELLS_ZH.md`

### Colab mixed-size one-command pipeline / Colab 混合尺寸一键脚本

> Note / 说明: if logs report timestamp gaps / multiple time segments, this is only a diagnostic warning about jumps in the CSV timeline. The current workflow still treats **one CSV as one complete scene/view trial** and does **not** auto-split the file into multiple trials.
>
> Recommended / 现推荐：use the research-bundle one-command entry below. If `scene_features.csv` is missing, the repo will auto-generate one from the scene folders, AOI JSON, and background images.

```bash
cd /content/eyetrack
python scripts/run_colab_one_command.py \
  --scenes_root_orig /content/drive/MyDrive/映射 \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv
```

Equivalent compatibility entry / 等价兼容入口：

```bash
cd /content/eyetrack
python scripts/run_colab_analysis2_pipeline.py \
  --scenes_root_orig /content/drive/MyDrive/映射 \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv
```

If you already finished the mixed-size batch + merged step and only want to optimize the merged folder:

```bash
cd /content/eyetrack
python scripts/optimize_merged_aoi_outputs.py \
  --merged_outdir "/content/drive/MyDrive/映射/AOI输出_xxx/输出结果_AOI_合并" \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv
```

### 1) Install dependencies / 安装依赖

Run the first cells in the notebook to install `requirements.txt`.
按 Notebook 从上到下运行到安装 `requirements.txt` 的 cell。

### 2) Single CSV (basic heatmap + scanpath) / 单文件快速出图

Upload one eye-tracking CSV, then run pipeline.
上传一个 CSV 后直接跑基础脚本。

Outputs / 输出：
- `outputs/quality_report.csv`
- `outputs/heatmap.png`
- `outputs/scanpath.png`

### 3) Batch heatmaps (no AOI) / 批处理热图（不做 AOI）

**Recommended**: zip all CSVs and upload once.
建议把所有 CSV 打包成一个 zip 再上传。

In Colab:

```python
from google.colab import files
files.upload()  # upload csvs.zip
```

```bash
!rm -rf batch_csvs
!mkdir -p batch_csvs
!unzip -q csvs.zip -d batch_csvs

!python scripts/batch_heatmap.py \
  --input_dir batch_csvs \
  --screen_w 1920 --screen_h 1080 \
  --outdir outputs_batch_heatmap
```

Outputs / 输出：
- `outputs_batch_heatmap/<file_stem>/heatmap.png`
- `outputs_batch_heatmap/batch_quality_report.csv`

#### (Optional) Overlay heatmap on background image /（可选）热图叠加到底图

Upload a background image (e.g. `scene.png`) and ensure its pixel size equals `screen_w/screen_h`.
上传底图（如 `scene.png`），并确保底图像素尺寸与 `screen_w/screen_h` 一致。

```bash
!python scripts/batch_heatmap.py \
  --input_dir batch_csvs \
  --screen_w 1920 --screen_h 1080 \
  --background_img scene.png \
  --outdir outputs_batch_heatmap
```

Extra output / 额外输出：
- `heatmap_overlay.png`

### 4) Grouped batch + difference plots / 分组批处理 + 差异图（SportFreq / Experience / 4-way）

Prepare a manifest `group_manifest.csv` with 3 columns:
准备一个三列表格：

```csv
name,SportFreq,Experience
Alice,High,Low
Bob,Low,High
```

Then run (all CSVs in one folder):
然后运行（所有 CSV 在同一目录）：

```bash
!python scripts/batch_heatmap_groups.py \
  --manifest group_manifest.csv \
  --csv_dir batch_csvs \
  --screen_w 1920 --screen_h 1080 \
  --outdir outputs_batch_groups
```

Key outputs / 主要输出：
- Individual / 每人：`outputs_batch_groups/individual/<name>/heatmap.png`
- Group densities / 分组汇总：`outputs_batch_groups/groups/**/heatmap_density.png`
- Binary diff / 二分差异图：
  - `outputs_batch_groups/compare/SportFreq_diff.png`
  - `outputs_batch_groups/compare/Experience_diff.png`
- 4-way overview / 四类对比：`outputs_batch_groups/compare/4way_grid.png`

#### (Optional) Overlay group heatmaps on background /（可选）分组汇总图叠加到底图

```bash
!python scripts/batch_heatmap_groups.py \
  --manifest group_manifest.csv \
  --csv_dir batch_csvs \
  --screen_w 1920 --screen_h 1080 \
  --background_img scene.png \
  --outdir outputs_batch_groups
```

Extra outputs / 额外输出：
- `outputs_batch_groups/groups/**/heatmap.png` (overlay when background is provided)
- `outputs_batch_groups/groups/**/heatmap_overlay.png`
- Compare plots also overlay background when `--background_img` is provided.

### 5) Download results / 下载结果

```bash
!zip -qr outputs_batch_groups.zip outputs_batch_groups
```

```python
from google.colab import files
files.download('outputs_batch_groups.zip')
```

## 5. End-to-End Workflow / 全流程操作

### One-command runner (optional) / 一键运行（可选）

#### Recommended Colab one-command entry / 推荐 Colab 一条命令入口

For the indoor-pingpong / multi-scene research-bundle workflow, the simplest Colab command is:

```bash
cd /content/eyetrack
python scripts/run_colab_one_command.py \
  --scenes_root_orig /content/drive/MyDrive/映射 \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv
```

Inputs expected under `--scenes_root_orig`:
- one folder per scene
- each scene folder contains the background image, AOI JSON, and all participant CSVs for that scene
- `group_manifest.csv` contains at least `name` (participant id), and usually `SportFreq` / `Experience`

Outputs:
- `research_bundle_YYYYMMDD_HHMMSS/task1/` organized AOI results + grouped summaries
- `research_bundle_YYYYMMDD_HHMMSS/task2/` allocation LMM results
- `research_bundle_YYYYMMDD_HHMMSS/task3/` merged analysis table + two-part model results
- `research_bundle_YYYYMMDD_HHMMSS/diagnostics/` overlap / valid-ratio / distribution diagnostics
- `research_bundle_YYYYMMDD_HHMMSS/scene_features_autogenerated.csv` when no manual `scene_features.csv` is supplied

How `scene_features.csv` works now:
- If you pass `--scene_features_csv`, the pipeline uses your file directly.
- If you do not pass it, the pipeline auto-generates `scene_features_autogenerated.csv` from scene folders.
- Auto-generation can derive structural columns like `scene_id`, `WWR`, `Complexity`, `round`, image size, AOI coverage, `table_density`, `occlusion_ratio`, `crowding_level`, and table-center geometry.
- Manual columns such as `illum_lux`, `noise_db`, or real-world meter distances are still optional user-provided enrichments, not required for the default one-command run.

#### Minimal 4-input bundle / 最小四输入一键入口

If your real input is only:
- a folder of eye-tracking CSV files / 眼动 CSV 文件夹
- `group_manifest.csv`
- one scene background image / 一张场景底图
- one scene AOI JSON / 一份场景 AOI JSON

use:

```bash
python scripts/run_minimal_aoi_bundle.py \
  --csv_dir /path/to/csv_folder \
  --group_manifest /path/to/group_manifest.csv \
  --scene_image /path/to/scene.png \
  --aoi_json /path/to/scene_aoi.json \
  --scene_id WWR45_C1 \
  --outdir outputs_minimal_bundle
```

What it does / 它会自动完成：
- detect scene size from the background image / 从底图自动读取尺寸
- stage the single scene into the repo's batch format / 自动整理成批处理所需目录结构
- run fixation-based AOI metrics / 使用 fixation 口径计算 AOI 指标
- export overlap / time-segment / exclusion diagnostics / 导出重叠、时间段、排除日志
- generate organized outputs under `optimized_outputs/` / 生成整理后的结果目录

This is the recommended entry when you do **not** have or want `scene_features.csv`.
当你**没有也不想依赖** `scene_features.csv` 时，推荐优先使用这个入口。

#### Single file mode / 单文件模式

If you already have `aoi.json` and raw CSV, you can run the full workflow with:

```bash
python scripts/run_all.py \
  --input_csv your.csv \
  --aoi_json aoi.json \
  --scene_features_csv templates/indoor_pingpong_scene_features_template.csv \
  --workdir outputs_run_all
```

`run_all.py` is mainly for the older full paper-oriented chain and will require `scene_features_csv`
if you keep merge/model/figure stages on.
`run_all.py` 更偏向旧版“论文全链条”入口；如果不跳过 merge/model/figure 阶段，它会要求 `scene_features_csv`。

You can skip stages with flags like `--skip_model` / `--skip_figures`.

#### Batch mode / 批处理模式（manifest）

If you have multiple participants/scenes, prepare a manifest like `templates/batch_manifest_template.csv` and run:

```bash
python scripts/run_all.py \
  --manifest templates/batch_manifest_template.csv \
  --workdir outputs_run_all_batch \
  --dwell_mode fixation
```

If you also want to apply screen/validity filtering in batch (optional):

```bash
python scripts/run_all.py \
  --manifest templates/batch_manifest_template.csv \
  --workdir outputs_run_all_batch \
  --batch_filter \
  --screen_w 1280 --screen_h 1440
```


### Step A. Setup Environment / 环境准备

```bash
cd /home/wannaqueen66/.openclaw/workspace/eyetrack
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 --version
```

### Step B. Run Basic Analysis / 运行基础分析

```bash
python scripts/run_pipeline.py \
  --input /home/wannaqueen66/raw_User1_260128181841_0208103256.csv \
  --outdir outputs
```

Generated files / 生成文件：

- `outputs/quality_report.csv`
- `outputs/heatmap.png`
- `outputs/scanpath.png`
- `outputs/aoi_metrics.csv` (demo indoor-pingpong rectangular AOI / 室内乒乓球示例矩形AOI)

### Step C. Draw AOIs in Browser / 网页框选AOI

AOI web labeling is now maintained in a standalone repo: `eyetrack-aoi`.
AOI 网页标注已迁移到独立仓库：`eyetrack-aoi`。

- GitHub: `https://github.com/wannaqueen66-create/eyetrack-aoi`
- Preferred: open your deployed Cloudflare Pages/Workers URL
- Or run locally from `eyetrack-aoi/public` (simple static server)

Workflow / 操作流程：

1. Upload **background scene image** (required) / 上传**场景底图**（必选）
2. (Optional) Upload gaze CSV overlay / 可选上传 gaze CSV 叠加点
3. Input AOI class name / 输入 AOI 类名（如 `pingpong_table`）
4. Draw one or more polygons for that class / 为该类绘制一个或多个多边形
5. Export `aoi.json` / 导出 `aoi.json`

### Step D. Compute AOI Metrics / 计算AOI指标

```bash
cd /home/wannaqueen66/.openclaw/workspace/eyetrack
python scripts/run_aoi_metrics.py \
  --csv /home/wannaqueen66/raw_User1_260128181841_0208103256.csv \
  --aoi /path/to/aoi.json \
  --outdir outputs \
  --dwell_mode fixation \
  --screen_w 1280 --screen_h 1440 --require_validity

# If your CSV is already cleaned (e.g., produced by your own preprocessing), you can skip filtering:
python scripts/run_aoi_metrics.py \
  --csv cleaned.csv \
  --aoi aoi.json \
  --outdir outputs \
  --dwell_mode fixation \
  --assume_clean
```

Output files / 输出文件：

- `outputs/aoi_metrics_by_polygon.csv`（每个子区域）
- `outputs/aoi_metrics_by_class.csv`（类别汇总，论文常用）

Note: canonical metric columns now use `FC / FFD / MFD / MPD / RFF / TFD / TFF`.
Legacy aliases such as `fixation_count / TTFF_ms / dwell_time_ms / RF` are still kept temporarily for backward compatibility.

**Group summaries (SportFreq / Experience)**
After you have `batch_aoi_metrics_by_class.csv`, you can summarize outcomes by groups (like heatmap grouping):

```bash
python scripts/summarize_aoi_groups.py \
  --aoi_class_csv outputs_aoi_all/batch_aoi_metrics_by_class.csv \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv \
  --outdir outputs_aoi_groups \
  --plots
```

Outputs:
- `outputs_aoi_groups/aoi_group_summary.csv` (visited rate + conditional TFF/TFD summaries)
- `outputs_aoi_groups/aoi_with_groups.csv` (analysis-ready merged long table)

**What are `aoi_overlays/` and `plots/`? (quick)**
- `outdir/aoi_overlays/<scene_id>.png`: AOI *definition* audit figures. Polygons from `aoi.json` are drawn on the background image to verify AOI location/shape and coordinate consistency. These figures do **not** include gaze/fixation points and do **not** represent group differences.
- `outdir/plots/*.png`: AOI *result* figures (group summaries). These plots visualize group-level outcomes such as `visited_rate`, and conditional `TFF` / `TFD` given `visited==1` (two-part reporting idea). Numeric labels on bars show the aggregated values (%, ms).

### Step E. Read Results / 解读结果

- `TFD`: total fixation duration in AOI (recommended: aggregate by fixation) / AOI 总注视时长（推荐按 fixation 去重聚合）
- `TFF`: time to first fixation / 首次注视时间
- `FC`: fixation count / 注视次数
- `FFD`: first fixation duration / 首次注视时长
- `MFD`: mean fixation duration / 平均注视时长
- `RFF`: re-fixation frequency / 重注视频率
- `MPD`: mean pupil diameter / 平均瞳孔直径
- `visited`: whether the AOI was visited in this trial/scene (1=yes, 0=no). If `visited==0`, then `TFF` is NaN by definition. / 本次试次/场景是否进入该 AOI（1=是，0=否）。当 `visited==0` 时，`TFF` 按定义为 NaN。
- `polygon_count`: number of polygons under class / 该类别下子区域数量

**New options (recommended)**
- `--point_source fixation`: use `Fixation Point X/Y` for AOI hit testing (aligns better with fixation-based dwell/TTFF)
- `--dwell_empty_as_zero`: set TFD=0.0 when visited==0 (keeps TFF as NaN)
- `--image_match error`: if aoi.json includes image width/height and you pass --screen_w/--screen_h, stop on mismatch (default)
- `--trial_start_ms` / `--trial_start_col`: control TFF baseline t0 (optional; default t0=min timestamp)
- `--time_segments {warn,error,ignore}`: detect timestamp discontinuities (multi-trial risk) and warn/error
- `--report_time_segments`: export `timestamp_segments_summary.csv` (per file in single-run; per participant×scene in batch)
- `--min_valid_ratio`: trial-level tracking-rate threshold; exports `exclusion_log.csv` / `batch_exclusion_log.csv` when set
- `--warn_class_overlap` (default on): warn if different AOI classes overlap in screen space
- `--report_class_overlap`: export overlap table (`aoi_class_overlap.csv` or `batch_aoi_class_overlap.csv`) including overlap ratios

**How to describe these checks in a paper (template)**
- *AOI size consistency*: We ensured AOI definitions were drawn on the same background image size as the eye-tracking coordinates (mismatched AOI image size vs. screen size was treated as an error).
- *TFF missingness*: If an AOI was not visited, `TFF` was undefined and recorded as missing (NaN); visit probability and conditional TFF were analyzed separately (two-part strategy).
- *Tracking-rate inclusion*: We computed a trial-level valid ratio based on screen bounds and (optionally) validity flags, and logged trial exclusions when valid_ratio fell below a pre-defined threshold.
- *Multi-trial protection*: We flagged potential multi-segment recordings by detecting timestamp discontinuities (negative jumps or large gaps) and reported a segment count summary.
- *AOI overlap*: We checked overlaps between AOI classes in screen space and reported overlap counts/ratios when present.

For reproducibility, AOI scripts write `run_config.json` into the output directory.

**AOI overlay figure (recommended for audit / paper appendix)**
- `--export_aoi_overlay`: export `aoi_overlay.png` (single-run) or per-scene overlays under `outdir/aoi_overlays/<scene_id>.png` (batch).

---

## 6. AOI Web Tool v3 Guide / AOI网页工具v3说明

> Source repo / 源仓库: `https://github.com/wannaqueen66-create/eyetrack-aoi`


**EN**

- Draw irregular AOIs with polygons
- Group multiple polygons into one class
- Drag vertices to edit
- Move selected polygon with `Shift + drag`
- Insert/delete vertex
- Zoom with wheel; pan with `Alt+drag` or middle mouse drag
- Auto-close polygon by clicking near first point
- Import/export `aoi.json`
- Toggle gaze overlay

**中文**

- 支持不规则多边形 AOI
- 同类支持多个分离区域
- 顶点可拖拽编辑
- `Shift + 拖动` 可整体移动选中 polygon
- 支持顶点插入/删除
- 滚轮缩放，`Alt+拖动` 或中键拖动平移
- 点击首点附近自动闭合
- 支持导入/导出 `aoi.json`
- 支持 gaze 点叠加显示开关

---

## 7. Input Data Requirements / 输入数据要求

Required columns / 必需字段（建议同名）：

- `Recording Time Stamp[ms]`
- `Gaze Point X[px]`
- `Gaze Point Y[px]`
- `Fixation Index`
- `Fixation Duration[ms]`

Recommended columns / 推荐字段：

- `Validity Left`
- `Validity Right`

If your column names are different, you have two options:

1) **Preferred**: edit the column mapping file:
   - `configs/columns_default.json`
   - add your exporter column names to the candidate list

2) (Legacy) directly change field names in:
   - `src/pipeline.py`
   - `src/aoi_metrics.py`

Example / 示例：

```json
{
  "Gaze Point X[px]": ["Gaze Point X[px]", "x"],
  "Gaze Point Y[px]": ["Gaze Point Y[px]", "y"]
}
```

You can also pass a custom mapping file at runtime:

```bash
python scripts/run_aoi_metrics.py --csv your.csv --aoi aoi.json --columns_map /path/to/columns.json
```

### When to use screen/validity filtering? / 什么时候需要 screen/validity 过滤？

- If your CSV contains many out-of-range gaze coordinates, set:
  - `--screen_w` / `--screen_h`
- If your exporter provides validity flags and you want strict cleaning, set:
  - `--require_validity`

If you already cleaned data via `scripts/run_pipeline.py`, you can skip these filters for AOI metrics.

---

## 8. FAQ / 常见问题

### Q1. `ModuleNotFoundError: pandas`
Install dependencies in venv:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Q2. Cannot open AOI web page / AOI网页打不开
- Recommended: use deployed Cloudflare URL from `eyetrack-aoi` (Pages/Workers)
- If running locally, serve from `eyetrack-aoi/public`:
  - `python3 -m http.server 8080`
- Check firewall/security group allows port 8080 when accessing via VPS IP

### Q3. AOI result seems wrong / AOI结果不对
- Check image resolution matches CSV coordinate system
- Verify AOIs are drawn on the exact mapping image
- Check outlier gaze points (negative or huge values)

### Q4. `TFF` is NaN
No gaze entered that AOI class in that trial/scene (`visited==0`). This is expected behavior.

Recommended reporting (paper-friendly):
- Report both TFF (conditioned on `visited==1`) AND the non-visit probability:
  - `p_not_visited = P(visited==0)` per condition × AOI
- For inferential stats, use a two-part approach:
  1) Model `visited` (binary) with a logit model (preferably GLMM with participant random intercept)
  2) Model `TFF` on the subset where `visited==1`

See: `scripts/summarize_aoi_visit_rate.py`

### (New) Diagnostics + two-part modeling helpers

- Distribution diagnostics: `scripts/aoi_distribution_diagnostics.py`
- Two-part modeling helper (visited + conditional TFF/TFD/FC): `scripts/model_aoi_two_part.py`

---

## 9. Paper-Oriented Next Steps / 论文向下一步

### Mixed-effects (LMM-style) modeling for AOI allocation / AOI 注意分配的混合效应建模

After you generate merged AOI outputs (especially `batch_aoi_metrics_by_class.csv`), you can run an LMM-style exploratory model to answer:
"Which factors (WWR × Complexity × Group) influence visual attention allocation across AOIs?"

This repo provides a pragmatic Python `statsmodels` implementation:

```bash
python scripts/model_aoi_lmm_allocation.py \
  --aoi_class_csv <OUT_MERGED>/batch_aoi_metrics_by_class.csv \
  --group_manifest <scenes_root>/group_manifest.csv \
  --outdir <OUT_MERGED>/outputs_aoi_lmm
```

It will fit models for BOTH population group variables:
- `Experience` (High/Low)
- `SportFreq` (High/Low)

Outputs (per group variable):
- `model_*.txt` (model summaries)
- `fixef_*.csv` (tidy fixed-effect tables)

#### Descriptive tables + interaction PNGs / 描述性汇总表 + 交互图

To get the journal-friendly "means by (WWR×Complexity) × Group × AOI" tables and interaction plots:

```bash
python scripts/summarize_aoi_by_condition_group.py \
  --aoi_class_csv <OUT_MERGED>/batch_aoi_metrics_by_class.csv \
  --group_manifest <scenes_root>/group_manifest.csv \
  --outdir <OUT_MERGED>/outputs_aoi_summary
```

It exports:
- `outputs_aoi_summary/tables/summary_<GroupVar>_<outcome>.csv`
- `outputs_aoi_summary/plots/plot_<GroupVar>_<outcome>.png`

> Note: binary/count outcomes are better modeled via GLMM in R (lme4/glmmTMB).
> The modeling script focuses on LMM-style exploratory analysis on transformed outcomes.

**EN**

1. Use semantic AOIs (sky, greenery, facade, road, entrance, signage)
2. Run multi-participant batch processing
3. Merge environmental predictors (GVI, enclosure, facade complexity)
4. Build mixed-effects models with participant random effect
5. Export publication-ready figures/tables

**中文**

1. 将 AOI 升级为语义类别（天空/绿化/立面/道路/入口/标识等）
2. 扩展到多被试批处理
3. 合并环境解释变量（绿视率、开敞度、立面复杂度等）
4. 构建混合效应模型（被试作随机效应）
5. 输出投稿级图表与结果表

---

## 10. Optimized Output Workflow / 优化后的输出流程

This workflow reorganizes AOI batch outputs into a structure that is easier for scene-based and participant-based analysis, and adds grouped summaries/plots.
该流程会把 AOI 批处理结果重组为更适合“按场景”和“按被试”分析的结构，并自动生成按人群分组的汇总表与图。

### Command / 命令

If your data is a **single scene** with one background image + one AOI JSON + one folder of participant CSVs,
prefer `scripts/run_minimal_aoi_bundle.py` first.
如果你的数据是**单一场景**：一张底图 + 一份 AOI JSON + 一个参与者 CSV 文件夹，优先用 `scripts/run_minimal_aoi_bundle.py`。

Option A (run optimize as a separate step):

```bash
python scripts/optimize_aoi_outputs.py \
  --aoi_class_csv /path/to/batch_aoi_metrics_by_class.csv \
  --aoi_polygon_csv /path/to/batch_aoi_metrics_by_polygon.csv \
  --group_manifest /path/to/group_manifest.csv \
  --group_id_col name \
  --outdir outputs_organized
```

Option B (single command integrated in batch):

```bash
python scripts/batch_aoi_metrics.py \
  --group_manifest /path/to/group_manifest.csv \
  --scenes_root /path/to/scenes_root \
  --outdir outputs_batch \
  --dwell_mode fixation --point_source fixation \
  --optimize_outputs
```

`--optimize_outputs` will automatically generate `optimized_outputs/` under `--outdir`.

### Output structure / 输出结构

```text
outputs_organized/
├─ by_scene/
│  └─ <scene_id>/participants/
│     ├─ <participant>_class.csv
│     └─ <participant>_polygon.csv
├─ by_participant/
│  └─ <participant_id>/
│     ├─ <scene_id>_class.csv
│     └─ <scene_id>_polygon.csv
└─ grouped/
   ├─ tables/
   │  ├─ summary_sportfreq_scene.csv
   │  ├─ summary_sportfreq_condition.csv
   │  ├─ summary_experience_scene.csv
   │  └─ summary_experience_condition.csv
   └─ plots/
      ├─ sportfreq_scene_visited_rate.png
      ├─ sportfreq_scene_TFF.png
      ├─ sportfreq_scene_TFD.png
      ├─ sportfreq_scene_FC.png
      ├─ sportfreq_scene_FFD.png
      ├─ sportfreq_scene_MFD.png
      ├─ sportfreq_scene_RFF.png
      ├─ sportfreq_scene_MPD.png
      ├─ sportfreq_condition_visited_rate.png
      ├─ sportfreq_condition_TFF.png
      ├─ sportfreq_condition_TFD.png
      ├─ sportfreq_condition_FC.png
      ├─ experience_scene_visited_rate.png
      ├─ experience_scene_TFF.png
      ├─ experience_scene_TFD.png
      ├─ experience_scene_FC.png
      ├─ experience_condition_visited_rate.png
      ├─ experience_condition_TFF.png
      ├─ experience_condition_TFD.png
      └─ experience_condition_FC.png
```

Notes / 说明：
- Grouped output currently includes only `SportFreq` and `Experience` (no 2×2 cross-group export).
- 当前分组仅输出 `SportFreq` 与 `Experience`，不导出 2×2 交叉分组。
- The optimizer now exports both **scene-level** and **condition-level** summaries/PNGs. Use `*_scene_*.png` when you need all 12 scene slots preserved; use `*_condition_*.png` when you intentionally want the repeated WWR×Complexity conditions collapsed across rounds.
- 优化脚本现在会同时导出 **scene-level** 与 **condition-level** 的表和 PNG。需要保留 12 个场景位置时请看 `*_scene_*.png`；如果你是有意把两轮重复的 WWR×Complexity 条件合并，再看 `*_condition_*.png`。
- If `group_manifest.csv` contains columns like `trial01_scene`, grouped PNGs will prefer those scene labels (e.g. `WWR45_C1`) and plot by trial order. Scene-level plots prepend round tags such as `R1 WWR45_C1` / `R2 WWR45_C1`, so repeated conditions from round 1 vs round 2 are visually distinct and never collapsed into one x-axis slot.
- 如果 `group_manifest.csv` 含有类似 `trial01_scene` 的列，分组 PNG 会优先使用这些场景标签（如 `WWR45_C1`），并按 trial 顺序绘制。scene-level 图会在标签前加上轮次前缀，如 `R1 WWR45_C1` / `R2 WWR45_C1`，从而明确区分第 1 轮与第 2 轮，避免在横轴上被错误合并。

## 11. Building and Environment Figure Style Parameters / B&E图形规范参数表

Use the following defaults to keep figures journal-friendly and consistent across scripts.
建议在各绘图脚本中统一采用以下参数，保持投稿图风格一致。

| Item | Recommended setting |
|---|---|
| Figure background | white |
| Style | clean axis, no top/right spines |
| Grid | y-axis only, light gray, alpha≈0.2 |
| Palette | low-saturation, color-blind-friendly (e.g., `#4C78A8`, `#F58518`, `#54A24B`, `#B279A2`) |
| Font size | 9–11 pt |
| Axis label size | 10 pt |
| Tick label size | 9 pt |
| Legend size | 9 pt |
| Line width | 1.0–1.5 |
| Bar alpha | 0.9–0.95 |
| Export DPI | 300 |
| Width (single-column style) | ~3.3–3.6 in |
| Width (double-column style) | ~7.0–7.2 in |
| Annotation | concise numeric labels only when needed |

Practical rule / 实操规则：
- Keep axis labels and units explicit (`ms`, `%`).
- Use consistent y-scale across comparable panels.
- Avoid decorative effects; prioritize readability and reproducibility.

## Chinese-only Version / 纯中文版本

See `README_zh.md` for full Chinese-only documentation.  
纯中文文档请查看 `README_zh.md`。
