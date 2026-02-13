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
└─ webapp/
   └─ index.html          AOI网页工具（v3）
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

启动 AOI 网页工具：

```bash
cd webapp
python3 -m http.server 8080
```

浏览器打开：`http://<VPS-IP>:8080`

---

## 4. 小白全流程

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

1. 上传映射图片
2. 可选上传 CSV（叠加 gaze 点辅助定位）
3. 输入 AOI 类名（如 `pingpong_table`）
4. 画一个或多个多边形
5. 导出 `aoi.json`

### 步骤 D：计算 AOI 指标

```bash
python scripts/run_aoi_metrics.py \
  --csv /home/wannaqueen66/raw_User1_260128181841_0208103256.csv \
  --aoi /path/to/aoi.json \
  --outdir outputs
```

输出：

- `outputs/aoi_metrics_by_polygon.csv`（每个子区域）
- `outputs/aoi_metrics_by_class.csv`（类别汇总）

### 步骤 E：看论文常用结果

重点看 `outputs/aoi_metrics_by_class.csv`：

- `dwell_time_ms`：该类 AOI 总停留时长
- `TTFF_ms`：首次注视时间
- `fixation_count`：注视次数
- `polygon_count`：该类包含的区域数量

---

## 5. AOI网页工具v3说明

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

如果你的列名不同，请修改：

- `src/pipeline.py`
- `src/aoi_metrics.py`

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

说明该 AOI 类没有被注视点命中，检查框选区域或数据质量。

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
