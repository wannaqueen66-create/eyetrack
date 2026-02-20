# eyetrack Colab 部署与使用流程（中文）

> 如果你不想打开 Notebook，而是希望把每段代码当作一个 cell 逐个复制粘贴运行：
> - `docs/COLAB_CELLS_ZH.md`

## 一键入口

- 标准版（中英混合）：
  https://colab.research.google.com/github/wannaqueen66-create/eyetrack/blob/main/notebooks/eyetrack_colab_quickstart.ipynb
- 纯中文逐 Cell 说明版：
  https://colab.research.google.com/github/wannaqueen66-create/eyetrack/blob/main/notebooks/eyetrack_colab_quickstart_zh.ipynb

---

## 使用步骤（单文件）

1. 打开上面的 Colab 链接
2. 依次运行每个代码块（从上到下）
3. 上传你的眼动 CSV（必须包含项目要求的列）
4. 自动生成基础输出：
   - `outputs/quality_report.csv`
   - `outputs/heatmap.png`
   - `outputs/scanpath.png`
5. （可选）上传 `aoi.json` 并运行 AOI 指标
6. 下载 `eyetrack_outputs.zip`

---

## 批处理：只生成热图（不做 AOI）

适用场景：你有很多个 CSV，想批量导出每个文件的 `heatmap.png`。

如果你还要做**分组对比**（比如 SportFreq 二分、Experience 二分、以及两者交叉得到 4 类人群），看本文档后面的「批处理：按人群分组汇总 + 差异图」。

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

## 批处理：按人群分组汇总 + 差异图（SportFreq 二分 / Experience 二分 / 4 类交叉）

适用场景：
- 你要每个人的单独热图
- 同时要按人群汇总热图（例如 SportFreq-High vs SportFreq-Low）
- 还要一张能直接看出组间差异的图（推荐使用**log-ratio 差异图**）

### Step 1. 准备 manifest（分组表）

在本仓库的模板基础上准备你的 manifest：
- 模板：`templates/group_manifest_template.csv`

必须列：
- `name`（或 `participant_id`）：被试姓名/ID
- `SportFreq`：High / Low
- `Experience`：High / Low

可选列：
- `csv_path`：该被试 CSV 路径

如果你不想在 manifest 里写 `csv_path`（比如所有 CSV 都在同一个文件夹），可以在运行脚本时传 `--csv_dir batch_csvs`，脚本会用 `participant_id` 去匹配对应 CSV 文件名。
### Step 2. 运行分组批处理脚本

```bash
!python scripts/batch_heatmap_groups.py \
  --manifest group_manifest.csv \
  --csv_dir batch_csvs \
  --screen_w 1920 --screen_h 1080 \
  --outdir outputs_batch_groups
```

如果你想把**各组汇总热图**也叠加到底图上，加 `--background_img`：

```bash
!python scripts/batch_heatmap_groups.py \
  --manifest group_manifest.csv \
  --csv_dir batch_csvs \
  --screen_w 1920 --screen_h 1080 \
  --background_img scene.png \
  --outdir outputs_batch_groups
```

输出会额外多一批：`heatmap_overlay.png`

输出结构（重点）：
- `outputs_batch_groups/individual/<participant_id>/heatmap.png`
- `outputs_batch_groups/groups/SportFreq-High/heatmap_density.png`
- `outputs_batch_groups/groups/SportFreq-Low/heatmap_density.png`
- `outputs_batch_groups/compare/SportFreq_diff.png`（High vs Low 差异图）
- `outputs_batch_groups/compare/Experience_diff.png`
- `outputs_batch_groups/compare/4way_grid.png`（四类共享色标的 2x2 汇总图）

### Step 3. 打包下载

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

