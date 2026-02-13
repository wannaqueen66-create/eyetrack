# eyetrack

> **Python-based eye-tracking analysis toolkit** for Building & Environment research.  
> 面向 Building and Environment 方向的眼动数据分析工具（Python），支持从原始 CSV 到 AOI 指标与可视化的一整套流程。

---

## Table of Contents / 目录

- [1. What this project does / 项目能做什么](#1-what-this-project-does--项目能做什么)
- [2. Who this is for / 适用人群](#2-who-this-is-for--适用人群)
- [3. Project structure / 项目结构](#3-project-structure--项目结构)
- [4. Quick start (5 mins) / 5分钟快速开始](#4-quick-start-5-mins--5分钟快速开始)
- [5. Full beginner workflow / 小白完整流程](#5-full-beginner-workflow--小白完整流程)
  - [Step A. Install / 安装环境](#step-a-install--安装环境)
  - [Step B. Run basic analysis / 运行基础分析](#step-b-run-basic-analysis--运行基础分析)
  - [Step C. Draw AOIs / 标注 AOI](#step-c-draw-aois--标注-aoi)
  - [Step D. Compute AOI metrics / 计算 AOI 指标](#step-d-compute-aoi-metrics--计算-aoi-指标)
  - [Step E. Interpret outputs / 结果解读](#step-e-interpret-outputs--结果解读)
- [6. AOI web tool guide (v3) / AOI网页工具说明（v3）](#6-aoi-web-tool-guide-v3--aoi网页工具说明v3)
- [7. Data format requirements / 数据格式要求](#7-data-format-requirements--数据格式要求)
- [8. FAQ / 常见问题](#8-faq--常见问题)
- [9. Next for paper-ready pipeline / 论文级下一步建议](#9-next-for-paper-ready-pipeline--论文级下一步建议)

---

## 1. What this project does / 项目能做什么

This repo helps you:

1. Clean eye-tracking CSV data
2. Generate visualizations (heatmap + scanpath)
3. Draw **irregular polygon AOIs** in browser
4. Support **multiple polygons under one AOI class** (e.g., 3 separate pingpong tables -> one class)
5. Export AOI metrics for papers:
   - Dwell time
   - TTFF (Time To First Fixation)
   - Fixation count

本仓库可以帮你：

1. 清洗眼动 CSV 数据
2. 生成可视化（热图 + scanpath）
3. 在网页中框选**不规则多边形 AOI**
4. 支持**同一类别多个分离区域**
5. 导出论文常用 AOI 指标（停留时长 / TTFF / 注视次数）

---

## 2. Who this is for / 适用人群

- Beginners with eye-tracking data from aseeStudio (or similar)
- Users who want Python workflow (not vendor software post-processing)
- Building & Environment researchers preparing figures/tables for papers

- 适合刚从 aseeStudio 导出数据、想转 Python 流水线的新手
- 适合 B&E 方向需要论文图表与 AOI 统计的研究者

---

## 3. Project structure / 项目结构

```text
eyetrack/
├─ data/
│  ├─ raw/                # 原始数据（建议不提交大文件）
│  └─ processed/          # 清洗后数据
├─ figures/               # 导出的图像
├─ outputs/               # 运行脚本后的输出（本地生成）
├─ scripts/
│  ├─ run_pipeline.py     # 基础分析：质检 + 热图 + scanpath +矩形AOI示例
│  └─ run_aoi_metrics.py  # 按 aoi.json 计算 polygon/class 指标
├─ src/
│  ├─ pipeline.py
│  └─ aoi_metrics.py
└─ webapp/
   └─ index.html          # AOI 标注网页（v3）
```

---

## 4. Quick start (5 mins) / 5分钟快速开始

```bash
cd /home/wannaqueen66/.openclaw/workspace/eyetrack
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) 基础分析
python scripts/run_pipeline.py \
  --input /home/wannaqueen66/raw_User1_260128181841_0208103256.csv \
  --outdir outputs

# 2) 开 AOI 网页标注
cd webapp && python3 -m http.server 8080
# 浏览器打开：http://<VPS-IP>:8080
```

---

## 5. Full beginner workflow / 小白完整流程

### Step A. Install / 安装环境

```bash
cd /home/wannaqueen66/.openclaw/workspace/eyetrack
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check Python version (recommended 3.9+):

```bash
python3 --version
```

---

### Step B. Run basic analysis / 运行基础分析

```bash
python scripts/run_pipeline.py \
  --input /home/wannaqueen66/raw_User1_260128181841_0208103256.csv \
  --outdir outputs
```

You will get / 你会得到：

- `outputs/quality_report.csv`
- `outputs/heatmap.png`
- `outputs/scanpath.png`
- `outputs/aoi_metrics.csv` (rectangular AOI demo)

---

### Step C. Draw AOIs / 标注 AOI

Start web app:

```bash
cd /home/wannaqueen66/.openclaw/workspace/eyetrack/webapp
python3 -m http.server 8080
```

Open in browser:

```text
http://<VPS-IP>:8080
```

Then:

1. Upload mapping image / 上传映射图片
2. (Optional) Upload gaze CSV for overlay / 可选上传 CSV 叠加注视点
3. Type class name (e.g., `pingpong_table`) / 输入类别名
4. Draw polygons / 绘制多边形
5. Export `aoi.json` / 导出 AOI 文件

---

### Step D. Compute AOI metrics / 计算 AOI 指标

```bash
cd /home/wannaqueen66/.openclaw/workspace/eyetrack
python scripts/run_aoi_metrics.py \
  --csv /home/wannaqueen66/raw_User1_260128181841_0208103256.csv \
  --aoi /path/to/aoi.json \
  --outdir outputs
```

Outputs / 输出：

- `outputs/aoi_metrics_by_polygon.csv` (each sub-region / 每个子区域)
- `outputs/aoi_metrics_by_class.csv` (class-level summary / 类别汇总)

---

### Step E. Interpret outputs / 结果解读

Key fields / 关键字段：

- `dwell_time_ms`: total fixation time in AOI / AOI 总停留时长
- `TTFF_ms`: time to first fixation / 首次注视时间
- `fixation_count`: number of fixations / 注视次数
- `polygon_count`: number of polygons under class / 该类下多边形数量

For paper tables, usually use class-level file:

- `aoi_metrics_by_class.csv`

---

## 6. AOI web tool guide (v3) / AOI网页工具说明（v3）

Implemented features / 已实现：

- Irregular polygon AOI / 不规则多边形 AOI
- Multiple polygons in one class / 同类多区域
- Vertex drag edit / 顶点拖拽编辑
- Polygon move (`Shift + drag`) / 整体拖动
- Insert/Delete vertex / 顶点插入与删除
- Auto-close snapping (click near first point) / 首点吸附自动闭合
- Pan/Zoom (`Alt+drag` or middle mouse drag, wheel zoom) / 平移缩放
- Import/Export `aoi.json` / AOI 导入导出
- Gaze point overlay toggle / 注视点叠加开关

---

## 7. Data format requirements / 数据格式要求

Your CSV should include these columns (exact names preferred):

- `Recording Time Stamp[ms]`
- `Gaze Point X[px]`
- `Gaze Point Y[px]`
- `Fixation Index`
- `Fixation Duration[ms]`

Recommended (for better cleaning):

- `Validity Left`
- `Validity Right`

If your column names differ, update script field names in:

- `src/pipeline.py`
- `src/aoi_metrics.py`

---

## 8. FAQ / 常见问题

### Q1: `ModuleNotFoundError: pandas`
Install dependencies in venv:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Q2: Browser cannot open `http://<VPS-IP>:8080`
- Check server is running in `webapp/` folder
- Check VPS firewall/security group allows 8080

### Q3: AOI result looks wrong
- Check image resolution matches CSV coordinate system
- Verify AOI polygons are drawn on the same mapping image
- Inspect outlier points (negative or too large coordinates)

### Q4: TTFF is NaN
This class may have no gaze points; verify AOI location and data quality.

---

## 9. Next for paper-ready pipeline / 论文级下一步建议

To make this publication-ready for Building & Environment:

1. Replace demo AOIs with semantic AOIs:
   - sky / greenery / façade / road / entrance / signage
2. Run multi-participant batch analysis
3. Merge environmental predictors (e.g., GVI, enclosure, facade complexity)
4. Fit mixed-effects models (participant as random effect)
5. Output figure templates for manuscript

如果你是第一次做论文，建议先按本文档完整跑通一次单被试，再扩展到多被试批处理。

---

## License

For research and educational use.
