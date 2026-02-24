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
  - dwell_time_ms
  - TTFF_ms
  - fixation_count

**中文**

- 眼动 CSV 数据清洗
- 热图与 scanpath 可视化
- 浏览器内多边形 AOI 标注
- 同一 AOI 类支持多个分离区域
- 指标输出支持“子区域级”和“类别级”：
  - dwell_time_ms（停留时长）
  - TTFF_ms（首次注视时间）
  - fixation_count（注视次数）

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

#### Single file mode / 单文件模式

If you already have `aoi.json` and raw CSV, you can run the full workflow with:

```bash
python scripts/run_all.py \
  --input_csv your.csv \
  --aoi_json aoi.json \
  --scene_features_csv templates/indoor_pingpong_scene_features_template.csv \
  --workdir outputs_run_all
```

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

### Step E. Read Results / 解读结果

- `dwell_time_ms`: total fixation time in AOI (recommended: aggregate by fixation) / AOI 总停留时长（推荐按 fixation 去重聚合）
- `TTFF_ms`: time to first fixation / 首次注视时间
- `fixation_count`: number of fixations / 注视次数
- `visited`: whether the AOI was visited in this trial/scene (1=yes, 0=no). If `visited==0`, then `TTFF_ms` is NaN by definition. / 本次试次/场景是否进入该 AOI（1=是，0=否）。当 `visited==0` 时，`TTFF_ms` 按定义为 NaN。
- `polygon_count`: number of polygons under class / 该类别下子区域数量

**New options (recommended)**
- `--point_source fixation`: use `Fixation Point X/Y` for AOI hit testing (aligns better with fixation-based dwell/TTFF)
- `--dwell_empty_as_zero`: set dwell_time_ms=0.0 when visited==0 (keeps TTFF as NaN)
- `--image_match error`: if aoi.json includes image width/height and you pass --screen_w/--screen_h, stop on mismatch (default)
- `--trial_start_ms` / `--trial_start_col`: control TTFF baseline t0 (optional; default t0=min timestamp)
- `--time_segments {warn,error,ignore}`: detect timestamp discontinuities (multi-trial risk) and warn/error
- `--report_time_segments`: export `timestamp_segments_summary.csv` (per file in single-run; per participant×scene in batch)
- `--min_valid_ratio`: trial-level tracking-rate threshold; exports `exclusion_log.csv` / `batch_exclusion_log.csv` when set
- `--warn_class_overlap` (default on): warn if different AOI classes overlap in screen space
- `--report_class_overlap`: export overlap table (`aoi_class_overlap.csv` or `batch_aoi_class_overlap.csv`) including overlap ratios

**How to describe these checks in a paper (template)**
- *AOI size consistency*: We ensured AOI definitions were drawn on the same background image size as the eye-tracking coordinates (mismatched AOI image size vs. screen size was treated as an error).
- *TTFF missingness*: If an AOI was not visited, `TTFF_ms` was undefined and recorded as missing (NaN); visit probability and conditional TTFF were analyzed separately (two-part strategy).
- *Tracking-rate inclusion*: We computed a trial-level valid ratio based on screen bounds and (optionally) validity flags, and logged trial exclusions when valid_ratio fell below a pre-defined threshold.
- *Multi-trial protection*: We flagged potential multi-segment recordings by detecting timestamp discontinuities (negative jumps or large gaps) and reported a segment count summary.
- *AOI overlap*: We checked overlaps between AOI classes in screen space and reported overlap counts/ratios when present.

For reproducibility, AOI scripts write `run_config.json` into the output directory.

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

### Q4. `TTFF_ms` is NaN
No gaze entered that AOI class in that trial/scene (`visited==0`). This is expected behavior.

Recommended reporting (paper-friendly):
- Report both TTFF (conditioned on `visited==1`) AND the non-visit probability:
  - `p_not_visited = P(visited==0)` per condition × AOI
- For inferential stats, use a two-part approach:
  1) Model `visited` (binary) with a logit model (preferably GLMM with participant random intercept)
  2) Model `TTFF_ms` on the subset where `visited==1`

See: `scripts/summarize_aoi_visit_rate.py`

### (New) Diagnostics + two-part modeling helpers

- Distribution diagnostics: `scripts/aoi_distribution_diagnostics.py`
- Two-part modeling helper (visited + conditional TTFF/dwell/count): `scripts/model_aoi_two_part.py`

---

## 9. Paper-Oriented Next Steps / 论文向下一步

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

## Chinese-only Version / 纯中文版本

See `README_zh.md` for full Chinese-only documentation.  
纯中文文档请查看 `README_zh.md`。
