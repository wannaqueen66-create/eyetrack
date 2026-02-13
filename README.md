# eyetrack

Building and Environment 方向的眼动数据分析项目（Python）。

## 目录结构

- `data/raw/` 原始数据（不入库大文件）
- `data/processed/` 清洗后数据
- `figures/` 导出的图（热图、scanpath、AOI 可视化）
- `src/` 核心分析代码
- `scripts/` 一键运行脚本
- `notebooks/` 探索性分析

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py \
  --input /home/wannaqueen66/raw_User1_260128181841_0208103256.csv \
  --outdir outputs
```

## 当前支持

- CSV 读取与字段清洗
- 基础质量报告（有效率、时长、坐标范围）
- 热图（kde）
- scanpath（抽样）
- AOI 指标（dwell time / fixation count / TTFF）

## 下一步（论文向）

- 建筑语义 AOI（天空/绿化/立面/道路/入口等）
- 多被试批处理
- 混合效应模型（被试随机效应）
- 论文图表模板（B&E 风格）
