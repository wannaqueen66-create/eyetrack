# Research Bundle Guide / eyetrack 主仓正式研究输出说明

`research_bundle` is the canonical research output bundle for the eye-tracking repository.
`research_bundle` 是 eyetrack 主仓的正式研究输出主线。

## Target contents / 目标内容

At minimum, the research bundle should include:
正式研究输出包至少应包含：

1. **Task1: descriptive AOI outputs**
   - organized outputs by scene / participant / group
   - WWR × Complexity × Group summaries
   - PNG plots

2. **Task2: allocation models**
   - LMM-style models across AOI class × WWR × Complexity × Group

3. **Task3: two-part models**
   - visited
   - TTFF | visited==1
   - TFD | visited==1
   - FC | visited==1

4. **Diagnostics**
   - distribution diagnostics
   - overlap reports
   - valid-ratio exclusion logs
   - timestamp-segment diagnostics

5. **Colab-compatible runner**
   - a script that can run the research bundle on Colab / Google Drive scene folders

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
python scripts/run_colab_analysis2_pipeline.py \
  --scenes_root_orig /content/drive/MyDrive/映射 \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv \
  --scene_features_csv /content/drive/MyDrive/映射/scene_features.csv
```

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

Use `TTFF` as the primary output name; keep legacy aliases like `TFF` / `TTFF_ms` / `dwell_time_ms` only for backward compatibility.
新输出统一使用 `TTFF` 作为主列名；`TFF` / `TTFF_ms` / `dwell_time_ms` 仅作为向后兼容别名保留。
