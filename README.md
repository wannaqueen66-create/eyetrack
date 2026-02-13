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

## 5. End-to-End Workflow / 全流程操作

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
- Or run locally from `eyetrack-aoi/public`

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
  --outdir outputs
```

Output files / 输出文件：

- `outputs/aoi_metrics_by_polygon.csv`（每个子区域）
- `outputs/aoi_metrics_by_class.csv`（类别汇总，论文常用）

### Step E. Read Results / 解读结果

- `dwell_time_ms`: total fixation time in AOI / AOI 总停留时长
- `TTFF_ms`: time to first fixation / 首次注视时间
- `fixation_count`: number of fixations / 注视次数
- `polygon_count`: number of polygons under class / 该类别下子区域数量

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

If your column names are different, update field names in:

- `src/pipeline.py`
- `src/aoi_metrics.py`

如果你的列名不同，请在上述两个脚本中替换字段名。

---

## 8. FAQ / 常见问题

### Q1. `ModuleNotFoundError: pandas`
Install dependencies in venv:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Q2. Cannot open web page on VPS / VPS网页打不开
- Ensure `python3 -m http.server 8080` is running under `webapp/`
- Check firewall/security group allows port 8080

### Q3. AOI result seems wrong / AOI结果不对
- Check image resolution matches CSV coordinate system
- Verify AOIs are drawn on the exact mapping image
- Check outlier gaze points (negative or huge values)

### Q4. `TTFF_ms` is NaN
No gaze entered that AOI class; check polygon location and data quality.

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
