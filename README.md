# eyetrack

Building and Environment 方向的眼动数据分析项目（Python）。

## 目录结构

- `data/raw/` 原始数据（不入库大文件）
- `data/processed/` 清洗后数据
- `figures/` 导出的图（热图、scanpath、AOI 可视化）
- `src/` 核心分析代码
- `scripts/` 一键运行脚本
- `webapp/` AOI 标注网页（不规则多边形 + 多区域同类）

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 基础分析（热图/scanpath/AOI矩形）
```bash
python scripts/run_pipeline.py \
  --input /home/wannaqueen66/raw_User1_260128181841_0208103256.csv \
  --outdir outputs
```

### AOI 标注网页（不规则多边形）
```bash
cd webapp
python3 -m http.server 8080
# 浏览器打开 http://<vps-ip>:8080
```

能力：
- 不规则 AOI：多边形点选闭合
- 同类多区域：一个类可挂多个 polygon
- 导出 `aoi.json`：`class_name -> [polygon1, polygon2, ...]`

### 按 AOI 类汇总指标（论文口径）
```bash
python scripts/run_aoi_metrics.py \
  --csv /home/wannaqueen66/raw_User1_260128181841_0208103256.csv \
  --aoi /path/to/aoi.json \
  --outdir outputs
```

输出：
- `outputs/aoi_metrics_by_polygon.csv`（每个子区域）
- `outputs/aoi_metrics_by_class.csv`（按类汇总）

核心字段：
- `dwell_time_ms`
- `TTFF_ms`
- `fixation_count`
