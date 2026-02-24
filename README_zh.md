# eyetrack（纯中文说明）

这是一个面向**室内乒乓球空间场景**的眼动分析项目。  
目标是：让你从 aseeStudio 导出的 CSV 出发，用 Python 完成数据清洗、可视化、AOI 标注与指标统计；并按 **Building and Environment（目标期刊）** 的论文规范组织结果。



> 说明：这里的 **Building and Environment** 指你的目标投稿期刊名称，不是场景类型。

---

## 目录

- [1. 能做什么](#1-能做什么)
- [2. 项目结构](#2-项目结构)
- [3. 快速开始](#3-快速开始)
- [4. 小白全流程](#4-小白全流程)
- [5. AOI网页工具v3说明](#5-aoi网页工具v3说明)
- [6. 输入数据要求](#6-输入数据要求)
- [7. 常见问题](#7-常见问题)
- [8. 论文向下一步](#8-论文向下一步)

---

## 1. 能做什么

- 清洗眼动 CSV 数据
- 生成热图和 scanpath
- 在网页中画不规则 AOI（多边形）
- 同一 AOI 类支持多个分离区域（如三张球桌同属 `pingpong_table`）
- 输出 AOI 指标：
  - 停留时长 `dwell_time_ms`
  - 首次注视时间 `TTFF_ms`
  - 注视次数 `fixation_count`

---

## 2. 项目结构

```text
eyetrack/
├─ data/
│  ├─ raw/                原始数据
│  └─ processed/          清洗后数据
├─ figures/               导出图像
├─ outputs/               脚本输出
├─ scripts/
│  ├─ run_pipeline.py     基础分析脚本
│  └─ run_aoi_metrics.py  AOI指标统计脚本
├─ src/
│  ├─ pipeline.py
│  └─ aoi_metrics.py
└─ AOI网页工具（已拆分）
   └─ https://github.com/wannaqueen66-create/eyetrack-aoi
```

---

## 3. 快速开始

```bash
cd /home/wannaqueen66/.openclaw/workspace/eyetrack
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_pipeline.py \
  --input /home/wannaqueen66/raw_User1_260128181841_0208103256.csv \
  --outdir outputs
```

启动独立 AOI 网页工具（已对接 eyetrack-aoi）：

- 仓库：`https://github.com/wannaqueen66-create/eyetrack-aoi`
- 推荐：Cloudflare Pages 一键部署后直接访问线上 URL
- 本地可选：

```bash
git clone https://github.com/wannaqueen66-create/eyetrack-aoi.git
cd eyetrack-aoi/public
python3 -m http.server 8080
```

浏览器打开：`http://<VPS-IP>:8080`

---


## Colab 部署（推荐给快速试跑）

- 直接打开：
  `https://colab.research.google.com/github/wannaqueen66-create/eyetrack/blob/main/notebooks/eyetrack_colab_quickstart.ipynb`
- 纯中文逐 Cell 版 Notebook：`https://colab.research.google.com/github/wannaqueen66-create/eyetrack/blob/main/notebooks/eyetrack_colab_quickstart_zh.ipynb`
- 详细中文说明（含：**批处理只出热图**）：`docs/COLAB_ZH.md`

## 4. 小白全流程

### 一键运行（可选）

#### 单文件模式

如果你已经有 `aoi.json` 和原始 CSV，可以用一个命令跑完整流程：

```bash
python scripts/run_all.py \
  --input_csv your.csv \
  --aoi_json aoi.json \
  --scene_features_csv templates/indoor_pingpong_scene_features_template.csv \
  --workdir outputs_run_all
```

你也可以通过 `--skip_model` / `--skip_figures` 等参数跳过部分步骤。

#### 批处理模式（manifest）

如果你有多被试/多场景，请先按 `templates/batch_manifest_template.csv` 准备 manifest，然后运行：

```bash
python scripts/run_all.py \
  --manifest templates/batch_manifest_template.csv \
  --workdir outputs_run_all_batch \
  --dwell_mode fixation
```

如需在批处理中也启用 screen/validity 过滤（可选）：

```bash
python scripts/run_all.py \
  --manifest templates/batch_manifest_template.csv \
  --workdir outputs_run_all_batch \
  --batch_filter \
  --screen_w 1280 --screen_h 1440
```


### 步骤 A：环境准备

```bash
cd /home/wannaqueen66/.openclaw/workspace/eyetrack
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 步骤 B：先跑基础分析

```bash
python scripts/run_pipeline.py \
  --input /home/wannaqueen66/raw_User1_260128181841_0208103256.csv \
  --outdir outputs
```

输出结果：

- `outputs/quality_report.csv`
- `outputs/heatmap.png`
- `outputs/scanpath.png`
- `outputs/aoi_metrics.csv`（室内乒乓球示例矩形AOI）

### 步骤 C：在网页里标注 AOI

AOI 标注网页已迁移到独立仓库：`eyetrack-aoi`
- GitHub：`https://github.com/wannaqueen66-create/eyetrack-aoi`
- 推荐使用 Cloudflare 部署后的线上地址

1. 上传**场景底图**（必选，且与 gaze 坐标对应）
2. 可选上传 CSV（叠加 gaze 点辅助定位）
3. 输入 AOI 类名（如 `pingpong_table`）
4. 画一个或多个多边形
5. 导出 `aoi.json`

### 步骤 D：计算 AOI 指标

```bash
python scripts/run_aoi_metrics.py \
  --csv /home/wannaqueen66/raw_User1_260128181841_0208103256.csv \
  --aoi /path/to/aoi.json \
  --outdir outputs \
  --dwell_mode fixation \
  --screen_w 1280 --screen_h 1440 --require_validity

# 如果你的 CSV 已经是清洗后的（例如你自己预处理导出的 clean 数据），可以跳过过滤：
python scripts/run_aoi_metrics.py \
  --csv cleaned.csv \
  --aoi aoi.json \
  --outdir outputs \
  --dwell_mode fixation \
  --assume_clean
```

输出：

- `outputs/aoi_metrics_by_polygon.csv`（每个子区域）
- `outputs/aoi_metrics_by_class.csv`（类别汇总）

**按组汇总（SportFreq / Experience）**
当你已经得到 `batch_aoi_metrics_by_class.csv` 后，可以像热力图一样按组汇总：

```bash
python scripts/summarize_aoi_groups.py \
  --aoi_class_csv outputs_aoi_all/batch_aoi_metrics_by_class.csv \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv \
  --outdir outputs_aoi_groups \
  --plots
```

输出：
- `outputs_aoi_groups/aoi_group_summary.csv`（visited 访问率 + 条件化 TTFF/dwell 等汇总）
- `outputs_aoi_groups/aoi_with_groups.csv`（用于建模的合并长表）

### 步骤 E：看论文常用结果

重点看 `outputs/aoi_metrics_by_class.csv`：

- `dwell_time_ms`：该类 AOI 总停留时长（推荐按 fixation 去重聚合，避免重复累计）
- `TTFF_ms`：首次注视时间
- `fixation_count`：注视次数
- `visited`：本次试次/场景是否进入该 AOI（1=是，0=否）。当 `visited==0` 时，`TTFF_ms` 按定义为 NaN。
- `polygon_count`：该类包含的区域数量

**新增选项（推荐）**
- `--point_source fixation`：用 `Fixation Point X/Y` 做 AOI 命中判定（与 fixation-based 指标更一致）
- `--dwell_empty_as_zero`：当 `visited==0` 时将 `dwell_time_ms` 记为 0.0（`TTFF_ms` 仍为 NaN）
- `--image_match error`：若 aoi.json 含底图宽高且你传入 --screen_w/--screen_h，则宽高不一致时直接报错停止（默认）
- `--trial_start_ms` / `--trial_start_col`：控制 TTFF 的基准 t0（可选；默认 t0=最小时间戳）
- `--time_segments {warn,error,ignore}`：检测时间戳断点/多段（多 trial 风险）并 warn/error
- `--report_time_segments`：导出 `timestamp_segments_summary.csv`（单文件=每个 CSV 一行；批处理=每个 participant×scene 一行）
- `--min_valid_ratio`：trial-level 追踪率阈值；设置后会输出 `exclusion_log.csv` / `batch_exclusion_log.csv`
- `--warn_class_overlap`（默认开启）：若不同 AOI 类在屏幕空间重叠，会输出警告提示
- `--report_class_overlap`：导出 overlap 表（`aoi_class_overlap.csv` 或 `batch_aoi_class_overlap.csv`，包含 overlap ratio）

**论文写法模板（可直接改动使用）**
- *AOI 尺寸一致性*：我们要求 AOI 标注所用底图的像素尺寸与眼动坐标系一致（aoi.json 记录的 image 宽高与 screen_w/screen_h 不一致时按错误处理）。
- *TTFF 缺失机制*：当 AOI 未被进入（visited=0）时，`TTFF_ms` 按定义不可得并记录为缺失（NaN）；后续采用 two-part 思路分别建模访问概率与条件化 TTFF。
- *追踪率纳入规则*：基于越界与（可选）Validity 字段计算 trial-level 的 valid_ratio，并在 valid_ratio 低于阈值时落盘 exclusion log（避免不同条件下追踪丢失造成系统性偏倚）。
- *多段记录保护*：通过检测时间戳断点（负跳变或大间隔）标记潜在多段/多 trial 的 CSV，并输出 segment 汇总表供排查。
- *AOI 重叠*：对不同 AOI 类在屏幕空间的重叠进行检测，并在存在时报告重叠的数量/比例。

为便于复现，AOI 脚本会在输出目录写入 `run_config.json`。

**AOI 可视化图（推荐用于审计/论文附录）**
- `--export_aoi_overlay`：导出 AOI overlay PNG（单文件输出 `aoi_overlay.png`；批处理输出到 `outdir/aoi_overlays/<scene_id>.png`）。

---

## 5. AOI网页工具v3说明

> 工具仓库：`https://github.com/wannaqueen66-create/eyetrack-aoi`


目前支持：

- 不规则多边形 AOI
- 同类多区域
- 顶点拖拽编辑
- `Shift + 拖动` 整体移动选中 polygon
- 顶点插入/删除
- 滚轮缩放 + `Alt+拖动`/中键拖动画布平移
- 点击首点附近自动闭合
- AOI 导入/导出 JSON
- gaze 点叠加显示开关

---

## 6. 输入数据要求

建议 CSV 至少有以下字段（列名尽量一致）：

- `Recording Time Stamp[ms]`
- `Gaze Point X[px]`
- `Gaze Point Y[px]`
- `Fixation Index`
- `Fixation Duration[ms]`

推荐再有：

- `Validity Left`
- `Validity Right`

如果你的列名不同，你有两种方式适配：

1）**推荐**：修改列名映射文件：
- `configs/columns_default.json`
- 把你导出 CSV 的列名追加到对应字段的候选列表里

2）（旧方式）直接在代码里改字段名：
- `src/pipeline.py`
- `src/aoi_metrics.py`

示例：

```json
{
  "Gaze Point X[px]": ["Gaze Point X[px]", "x"],
  "Gaze Point Y[px]": ["Gaze Point Y[px]", "y"]
}
```

也支持运行时指定自定义映射文件：

```bash
python scripts/run_aoi_metrics.py --csv your.csv --aoi aoi.json --columns_map /path/to/columns.json
```

### 什么时候需要 screen/validity 过滤？

- 如果你的 CSV 有很多超出屏幕范围的 gaze 坐标，建议加：
  - `--screen_w` / `--screen_h`
- 如果你的数据有 validity 标记并且你想严格过滤，建议加：
  - `--require_validity`

如果你已经用 `scripts/run_pipeline.py` 做过清洗，那么 AOI 指标统计这一步通常可以不再重复过滤。

---

## 7. 常见问题

### 1）报错 `ModuleNotFoundError: pandas`

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2）VPS 上网页打不开

- 确认在 `webapp` 目录启动了 `python3 -m http.server 8080`
- 确认防火墙/安全组已放行 8080

### 3）AOI 统计不合理

- 检查映射图片分辨率是否和 gaze 坐标系一致
- 确认 AOI 是在正确的映射底图上画的
- 检查 gaze 坐标是否有大量异常值

### 4）`TTFF_ms` 为 NaN

- 原因：该试次/场景中该 AOI 没有被注视到（`visited==0`），这是**预期行为**；当然数据质量或 AOI 标注偏移也可能导致“看起来不合理的未注视”。
- 建议论文报告方式（更强、更可解释）：
  1) 在 `visited==1` 的子样本上报告/建模 `TTFF_ms`
  2) 同时报告“未进入比例”`p_not_visited = P(visited==0)`（每个条件×AOI）
  3) 推断统计建议用“两部模型”：
     - 第一步：对 `visited` 做二项 logit（最好用 GLMM：被试随机截距）
     - 第二步：对 `TTFF_ms`（仅 visited==1）做线性混合/或 log 变换后建模

可用脚本：`scripts/summarize_aoi_visit_rate.py`

补充脚本（分布诊断 + 两部建模辅助）：
- 分布诊断：`scripts/aoi_distribution_diagnostics.py`
- 两部建模辅助（visited + 条件化 TTFF/dwell/count）：`scripts/model_aoi_two_part.py`

---

## 8. 论文向下一步

建议按 B&E 论文标准继续：

1. AOI 语义化：天空/绿化/立面/道路/入口/标识等
2. 多被试批处理
3. 合并环境解释变量（绿视率、开敞度、立面复杂度等）
4. 构建混合效应模型（被试随机效应）
5. 生成投稿图表模板

---

## 9. 室内乒乓球场景：已补齐的进阶模块

你本次项目是室内乒乓球空间，已新增以下脚本（就是之前 roadmap 里没做完的部分）：

1. **多被试批处理 AOI 指标**：`scripts/batch_aoi_metrics.py`
2. **场景变量合并**：`scripts/merge_scene_features.py`
3. **混合效应模型（被试随机效应）**：`scripts/mixed_effects_indoor_pingpong.py`
4. **论文图表导出**：`scripts/paper_figures_indoor_pingpong.py`

新增模板：
- `templates/batch_manifest_template.csv`
- `templates/indoor_pingpong_scene_features_template.csv`
- `configs/indoor_pingpong_aoi_classes.json`

### 最短可执行路径

```bash
# 1) 批量 AOI 指标（先把 manifest 填好）
python scripts/batch_aoi_metrics.py \
  --manifest templates/batch_manifest_template.csv \
  --outdir outputs_batch

# 2) 合并场景变量（先把 scene features 模板填好）
python scripts/merge_scene_features.py \
  --aoi_class_csv outputs_batch/batch_aoi_metrics_by_class.csv \
  --scene_features_csv templates/indoor_pingpong_scene_features_template.csv \
  --out_csv outputs/analysis_table.csv

# 3) 跑混合效应模型
python scripts/mixed_effects_indoor_pingpong.py \
  --analysis_csv outputs/analysis_table.csv \
  --outdir outputs_model

# 4) 导出论文图表
python scripts/paper_figures_indoor_pingpong.py \
  --analysis_csv outputs/analysis_table.csv \
  --outdir figures_paper
```
