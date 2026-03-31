# eyetrack Colab 部署与使用流程（中文）

> 如果你不想打开 Notebook，而是希望把每段代码当作一个 cell 逐个复制粘贴运行：
> - `docs/COLAB_CELLS_ZH.md`

## 一键入口

- 标准版（中英混合）：
  https://colab.research.google.com/github/wannaqueen66-create/eyetrack/blob/main/notebooks/eyetrack_colab_quickstart.ipynb
- 纯中文逐 Cell 说明版：
  https://colab.research.google.com/github/wannaqueen66-create/eyetrack/blob/main/notebooks/eyetrack_colab_quickstart_zh.ipynb

### 当前 `main` 分支：最短全量主线命令

如果你的目标是像 `raw` 分支那样，**用尽量少的几条 Colab 命令把当前 `main` 主线整套跑完**，直接用下面这组：

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!rm -rf /content/eyetrack
!git clone --branch main https://github.com/wannaqueen66-create/eyetrack.git /content/eyetrack
%cd /content/eyetrack
!pip -q install -r requirements.txt
!python scripts/run_colab_one_command.py \
  --scenes_root_orig /content/drive/MyDrive/映射 \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv
```

运行完成后，会在：

```text
/content/drive/MyDrive/映射/研究输出_YYYYMMDD_HHMMSS/
```

生成当前 `main` 主线的全量结果，包括：
- `00_全样本_AllSample`
- `01_QC后_AfterQC`
- 各自内部的描述性分析 / 显著性分析 / 附录输出

如果你已经手动 clone 过仓库，也可以直接只跑主命令：

```bash
cd /content/eyetrack
python scripts/run_colab_one_command.py \
  --scenes_root_orig /content/drive/MyDrive/映射 \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv
```

---

## 使用步骤（单文件）

1. 打开上面的 Colab 链接
2. 依次运行每个代码块（从上到下）
3. 现在脚本默认输出 PNG 标题为英文（ASCII-only），一般不会再触发中文字体缺失警告。
4. 最新 AOI / TTFF 流程会优先使用 CSV 中的 `Video Time[HH:mm:ss.ms]` 作为 TTFF 主时间轴，并结合 `Time of Day[HH:mm:ss.ms]` 自动检测 segment / gap / reset；结果表里会新增 `ttff_source / segment_count / ttff_warning / ttff_qc_status` 等 QC 字段，便于排查被整份 CSV 首行污染的情况。

如果你确实需要标题显示中文姓名，可以在 Colab 安装中文字体（避免中文标题刷 `Glyph missing` 警告）：
- `apt-get install -y fonts-noto-cjk`
并在脚本参数中使用：`--title_mode raw`
4. 上传你的眼动 CSV（必须包含项目要求的列）
4. 自动生成基础输出：
   - `outputs/quality_report.csv`
   - `outputs/heatmap.png`
   - `outputs/scanpath.png`
5. （可选）上传 `aoi.json` 并运行 AOI 指标
6. 下载 `eyetrack_outputs.zip`

---

## 批处理：只生成热图（不做 AOI）

适用场景：你有很多个 CSV，想批量导出每个文件的 `heatmap.png`。

如果你还要做**分组对比**（按 Experience 分组），看本文档后面的「批处理：按 Experience 分组汇总 + 对比图」。

### Step 1. 准备并上传 zip

把多个 CSV 放到同一个文件夹，压缩为 `csvs.zip`（内部可包含子文件夹）。

在 Colab 运行：

```python
from google.colab import files
files.upload()  # 上传 csvs.zip
```

解压：

```bash
!rm -rf batch_csvs
!mkdir -p batch_csvs
!unzip -q csvs.zip -d batch_csvs
```

### Step 2. 运行批处理脚本

在 Colab 设置你的屏幕分辨率（按实验屏幕像素改）：

```bash
!python scripts/batch_heatmap.py \
  --input_dir batch_csvs \
  --screen_w 1920 --screen_h 1080 \
  --outdir outputs_batch_heatmap
```

如果你想把热图**叠加到底图**（场景截图/首帧等），加 `--background_img`：

```bash
!python scripts/batch_heatmap.py \
  --input_dir batch_csvs \
  --screen_w 1920 --screen_h 1080 \
  --background_img scene.png \
  --outdir outputs_batch_heatmap
```

输出会额外多一个：`heatmap_overlay.png`

> 说明：批处理默认不强制 validity（更鲁棒）；如果你明确需要 validity 过滤，可以加 `--require_validity`。

输出：
- `outputs_batch_heatmap/<文件名>/heatmap.png`
- `outputs_batch_heatmap/batch_quality_report.csv`（汇总每个文件的清洗后有效比例/时长等；失败的会写 error）

### Step 3. 打包下载

```bash
!zip -qr outputs_batch_heatmap.zip outputs_batch_heatmap
```

```python
from google.colab import files
files.download('outputs_batch_heatmap.zip')
```

---

## 批处理：按 Experience 分组汇总 + 对比图

适用场景：
- 你要每个人的单独热图
- 要全体汇总热图（Overall）
- 要按 Experience 分组汇总（High vs Low）
- 要一张能直接看出组间差异的对比图

> 注意：当前版本默认使用 **fixation 点 + fixation duration 加权**（与 Tobii Pro Lab 默认一致）。如果你想改回 gaze 点，可加 `--point_source gaze --weight none`。

### Step 1. 准备 manifest（分组表）

在本仓库的模板基础上准备你的 manifest：
- 模板：`templates/group_manifest_template.csv`

必须列：
- `name`（或 `participant_id`）：被试姓名/ID
- `Experience`：High / Low

可选列：
- `csv_path`：该被试 CSV 路径
- `SportFreq`：存在也可以，但热力图脚本不再使用

如果你不想在 manifest 里写 `csv_path`（比如所有 CSV 都在同一个文件夹），可以在运行脚本时传 `--csv_dir batch_csvs`，脚本会用 `participant_id` 去匹配对应 CSV 文件名。

### Step 2. 运行分组批处理脚本

```bash
!python scripts/batch_heatmap_groups.py \
  --manifest group_manifest.csv \
  --csv_dir batch_csvs \
  --screen_w 1920 --screen_h 1080 \
  --background_img scene.png \
  --outdir outputs_batch_groups
```

输出结构：
- `outputs_batch_groups/individual/<name>/heatmap.png`（叠底图）
- `outputs_batch_groups/individual/<name>/heatmap_density.png`
- `outputs_batch_groups/groups/Overall/heatmap.png`
- `outputs_batch_groups/groups/Experience-High/heatmap.png`
- `outputs_batch_groups/groups/Experience-Low/heatmap.png`
- `outputs_batch_groups/compare/Experience_comparison.png`（三联对比图）
- `outputs_batch_groups/participants_summary.csv`

### Step 3. 多场景批处理（按场景分文件夹）

如果你有多个场景（例如 6 个或 12 个），每个场景文件夹里包含该场景的底图 + 所有人在该场景下的 CSV：

```bash
!python scripts/batch_heatmap_groups_scenes.py \
  --manifest group_manifest.csv \
  --scenes_root scenes \
  --screen_w 1748 --screen_h 2064 \
  --outdir outputs_by_scene
```

输出结构：
- `outputs_by_scene/<scene_name>/individual/...`
- `outputs_by_scene/<scene_name>/groups/Overall/...`
- `outputs_by_scene/<scene_name>/groups/Experience-High/...`
- `outputs_by_scene/<scene_name>/groups/Experience-Low/...`
- `outputs_by_scene/<scene_name>/compare/Experience_comparison.png`

### Step 4. 拼 6 场景投稿图

跑完多场景后，可以用 `compose_scene_panel.py` 把 6 个场景的 Overall 热力图拼成一张 3×2 投稿图：

```bash
!python scripts/compose_scene_panel.py \
  --scene_dirs outputs_by_scene \
  --outfile figures/Fig_heatmap_6scenes.png \
  --title "Fig. X. Fixation heatmaps across six visual conditions" \
  --note "Note: Heatmaps represent fixation-duration-weighted density across all participants. Warmer colors indicate higher visual attention concentration."
```

排列顺序：
```
(a) C0–WWR15   (b) C0–WWR45   (c) C0–WWR75
(d) C1–WWR15   (e) C1–WWR45   (f) C1–WWR75
```

脚本会自动根据文件夹名匹配场景位置，并去除黑边。

### Step 5. 打包下载

```bash
!zip -qr outputs_batch_groups.zip outputs_batch_groups
```

```python
from google.colab import files
files.download('outputs_batch_groups.zip')
```

---

## AOI 文件来源

`aoi.json` 建议在独立仓库网页里制作：

- https://github.com/wannaqueen66-create/eyetrack-aoi

---

## 常见问题

1. **上传后提示列名不匹配**
   - 检查 CSV 是否包含：
     - `Gaze Point X[px]`
     - `Gaze Point Y[px]`
     - `Recording Time Stamp[ms]`

2. **AOI 指标为空**
   - 检查 AOI 是否在正确底图上标注
   - 检查 gaze 坐标与底图像素坐标是否同一体系

3. **运行超时或中断**
   - Colab 会话断开后需重新运行前置格
