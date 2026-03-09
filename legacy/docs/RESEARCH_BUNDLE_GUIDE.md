# Research Output Guide / eyetrack 主仓正式研究输出说明

This file is the preferred guide for the canonical `研究输出_时间戳` output.
本文件是 `研究输出_时间戳` 正式研究输出主线的首选说明。

See also / 另见：
- `docs/METRICS_SPEC.md`
- `scripts/run_analysis2.py` (compatibility entry; now writes the unified research-output structure)
- `scripts/run_colab_analysis2_pipeline.py` (compatibility entry for Colab)
- `scripts/run_colab_one_command.py` (recommended Colab one-command entry)

## Required content / 必备内容

- `00_AOI原始批处理/`: raw AOI batch tables / overlays / merged batch exports
- `01_AOI与描述统计/`: organized outputs + grouped descriptive summaries
- `02_LMM模型/`: AOI allocation models + explanatory visuals
- `03_TwoPart模型/`: two-part models (when scene features are available)
- `04_诊断信息/`
- `05_colab说明/`
- `06_说明与报告/`
- `README_研究输出说明.txt`

## Recommended local runner / 推荐本地入口

```bash
python scripts/run_analysis2.py \
  --group_manifest /path/to/group_manifest.csv \
  --scenes_root /path/to/scenes_root \
  --scene_features_csv /path/to/scene_features.csv
```

## Recommended Colab runner / 推荐 Colab 入口

```bash
cd /content/eyetrack
python scripts/run_colab_one_command.py \
  --scenes_root_orig /content/drive/MyDrive/映射 \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv
```

## Output reading order / 输出阅读顺序

1. `研究输出_YYYYMMDD_HHMMSS/01_AOI与描述统计/`
2. `研究输出_YYYYMMDD_HHMMSS/02_LMM模型/`
3. `研究输出_YYYYMMDD_HHMMSS/03_TwoPart模型/`
4. `研究输出_YYYYMMDD_HHMMSS/04_诊断信息/`
5. `研究输出_YYYYMMDD_HHMMSS/00_AOI原始批处理/`

If you only need AOI-only mixed-size batch outputs, use `scripts/run_colab_aoi_pipeline.py`.
That script now also writes into the same naming family: `研究输出_AOI批处理_YYYYMMDD_HHMMSS/`.
如果你只需要 AOI-only 的 mixed-size 批处理，可用 `scripts/run_colab_aoi_pipeline.py`；
该脚本现在也统一写入同一命名家族：`研究输出_AOI批处理_YYYYMMDD_HHMMSS/`。
