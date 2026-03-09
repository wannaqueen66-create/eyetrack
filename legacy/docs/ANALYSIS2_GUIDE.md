# Research Output Guide / eyetrack 主仓正式研究输出说明

`研究输出_时间戳` is the canonical research output bundle for the eye-tracking repository.
`研究输出_时间戳` 是 eyetrack 主仓的正式研究输出主线。

## Target contents / 目标内容

At minimum, the research-output bundle should include:
正式研究输出包至少应包含：

1. **00_AOI原始批处理**
   - raw batch AOI tables
   - overlays
   - merged AOI exports for mixed-size runs

2. **01_AOI与描述统计**
   - organized outputs by scene / participant / group
   - WWR × Complexity × Group summaries
   - PNG plots

3. **02_LMM模型**
   - LMM-style models across AOI class × WWR × Complexity × Group
   - explanatory allocation plots

4. **03_TwoPart模型**
   - visited
   - TTFF | visited==1
   - TFD | visited==1
   - FC | visited==1

5. **04_诊断信息**
   - distribution diagnostics
   - overlap reports
   - valid-ratio exclusion logs
   - timestamp-segment diagnostics

6. **Colab-compatible runner / Colab 入口**
   - a script that can run the research output on Colab / Google Drive scene folders

## Official runners / 官方入口

### Local / VPS
```bash
python scripts/run_analysis2.py \
  --group_manifest /path/to/group_manifest.csv \
  --scenes_root /path/to/scenes_root \
  --scene_features_csv /path/to/scene_features.csv
```

### Colab
```bash
cd /content/eyetrack
python scripts/run_colab_one_command.py \
  --scenes_root_orig /content/drive/MyDrive/映射 \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv
```

### AOI-only Colab
```bash
cd /content/eyetrack
python scripts/run_colab_aoi_pipeline.py \
  --scenes_root_orig /content/drive/MyDrive/映射 \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv
```

## Naming policy / 命名策略

- Full one-command research runs now write to `研究输出_YYYYMMDD_HHMMSS/`
- AOI-only mixed-size runs now write to `研究输出_AOI批处理_YYYYMMDD_HHMMSS/`
- Old names like `research_bundle_*` / `AOI输出_*` remain legacy concepts only and are no longer the recommended primary output path

## Default metric naming / 默认指标命名

Use canonical abbreviations:
统一采用以下缩写：

- FC
- FFD
- MFD
- MPD
- RFF
- TFD
- TTFF

Use `TTFF` as the primary output name; keep legacy aliases only for backward compatibility.
新输出统一使用 `TTFF` 作为主列名；旧别名仅作向后兼容保留。
