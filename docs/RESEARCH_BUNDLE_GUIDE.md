# Research Bundle Guide / eyetrack 主仓正式研究输出说明

This file is the preferred guide for the canonical `research_bundle` output.
本文件是 `research_bundle` 正式研究输出主线的首选说明。

See also / 另见：
- `docs/METRICS_SPEC.md`
- `scripts/run_analysis2.py` (compatibility entry; now outputs `research_bundle` by default)
- `scripts/run_colab_analysis2_pipeline.py` (compatibility entry for Colab)

## Required content / 必备内容

- Task1: organized outputs + grouped descriptive summaries
- Task2: AOI allocation models
- Task3: two-part models (when scene features are available)
- diagnostics/
- reports/
- colab/

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
python scripts/run_colab_analysis2_pipeline.py \
  --scenes_root_orig /content/drive/MyDrive/映射 \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv \
  --scene_features_csv /content/drive/MyDrive/映射/scene_features.csv
```
