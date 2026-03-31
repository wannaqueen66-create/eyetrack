# eyetrack Colab 逐 Cell 复制版（最新）

> 目的：你不想打开现成 Notebook，而是希望把每段代码当作一个 Colab cell 逐个复制粘贴运行。
>
> 本文覆盖：
> - 批处理热图（不做 AOI）
> - 按 Experience 分组批处理
> - Experience 对比图
> - 热图叠加场景底图（overlay）
> - 6 场景拼合投稿图

> **当前默认行为**：热力图默认使用 **fixation 点 + fixation duration 加权 + Tobii-like 色带**，
> 与 Tobii Pro Lab 导出的热力图风格对齐。如需改回原始 gaze 点，可加 `--point_source gaze --weight none`。

---

## 你需要提前准备并上传到 Colab 的文件

- `csvs.zip`：眼动 CSV 数据
  - 旧版结构：所有 CSV 都在同一个文件夹（**每人 1 个 CSV**）
  - **新版结构（推荐/常见）**：按场景分文件夹（例如 6 个场景文件夹）；每个场景文件夹内包含该场景下所有人的 CSV（每人 1 个）
- `group_manifest.csv`：表头至少包含 `name,Experience`（每人 1 行）
- 场景底图（png/jpg）：
  - 旧版结构：所有人同一场景一张底图
  - 新版结构：每个场景文件夹内放该场景底图（与该场景 CSV 坐标同分辨率）

---

## Cell 0（可选）：确认当前目录

```bash
!pwd
!ls -la
```

---

## Cell 1：拉取最新仓库代码（main）

```bash
!rm -rf eyetrack
!git clone https://github.com/wannaqueen66-create/eyetrack.git
%cd eyetrack
!git log -1 --oneline
```

---

## Cell 2：安装依赖

```bash
!pip -q install -r requirements.txt
```

### （可选）Cell 2.1：安装中文字体（仅当你想在 PNG 标题里显示中文姓名）

> 脚本默认 `--title_mode english`（英文标题），一般不会刷 Glyph warning。
> 只有当你使用 `--title_mode raw` 想让标题显示中文姓名时，才需要装字体。

```bash
!apt-get -qq update
!apt-get -qq install -y fonts-noto-cjk
```

---

## Cell 3：上传文件（csvs.zip / group_manifest.csv）

```python
from google.colab import files
files.upload()
```

---

## Cell 4：解压 CSV

### 4A) 旧版结构：解压到同一文件夹 batch_csvs/

```bash
!rm -rf batch_csvs
!mkdir -p batch_csvs
!unzip -q csvs.zip -d batch_csvs

# quick check
!find batch_csvs -maxdepth 2 -type f -name "*.csv" | head
```

### 4B) 新版结构（推荐）：解压为"按场景分文件夹" scenes/

> 目标结构：
> - scenes/<scene_1>/*.csv + (scene_1).png
> - scenes/<scene_2>/*.csv + (scene_2).png
> - ...

```bash
!rm -rf scenes
!mkdir -p scenes
!unzip -q csvs.zip -d scenes

# quick check: list scene folders
!find scenes -maxdepth 1 -mindepth 1 -type d | head

# quick check: show some csv
!find scenes -maxdepth 2 -type f -name "*.csv" | head

# quick check: show some images
!find scenes -maxdepth 2 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | head
```

---

## Cell 5：检查底图分辨率（用于 screen_w/screen_h）

如果你上传的是 `scene.png`：

```python
from PIL import Image
Image.open('scene.png').size  # (width, height)
```

如果是 `scene.jpg`：

```python
from PIL import Image
Image.open('scene.jpg').size
```

---

## Cell 6：运行（分组批处理 + 对比图 + 叠底图）

把 `--screen_w/--screen_h` 改成 Cell 5 得到的分辨率。

### 6A) 旧版结构（单文件夹 CSV + 单张底图）

```bash
!python scripts/batch_heatmap_groups.py \
  --manifest group_manifest.csv \
  --csv_dir batch_csvs \
  --screen_w 1920 --screen_h 1080 \
  --background_img scene.png \
  --outdir outputs_batch_groups
```

### 6B) 新版结构（按场景分文件夹：一次跑完所有场景）

> 前提：每个场景文件夹里都放了该场景底图（png/jpg）

```bash
!python scripts/batch_heatmap_groups_scenes.py \
  --manifest group_manifest.csv \
  --scenes_root scenes \
  --screen_w 1748 --screen_h 2064 \
  --outdir outputs_by_scene
```

输出结构示例：
- `outputs_by_scene/<scene_name>/individual/...`
- `outputs_by_scene/<scene_name>/groups/Overall/...`
- `outputs_by_scene/<scene_name>/groups/Experience-High/...`
- `outputs_by_scene/<scene_name>/groups/Experience-Low/...`
- `outputs_by_scene/<scene_name>/compare/Experience_comparison.png`

---

## Cell 7：拼 6 场景投稿图

跑完多场景后，用 `compose_scene_panel.py` 把 6 个场景的 Overall 热力图拼成 3×2 投稿图：

```bash
!python scripts/compose_scene_panel.py \
  --scene_dirs outputs_by_scene \
  --outfile figures/Fig_heatmap_6scenes.png \
  --title "Fig. X. Fixation heatmaps across six visual conditions" \
  --note "Note: Heatmaps represent fixation-duration-weighted density across all participants."
```

排列：
```
(a) C0–WWR15   (b) C0–WWR45   (c) C0–WWR75
(d) C1–WWR15   (e) C1–WWR45   (f) C1–WWR75
```

---

## Cell 8：快速定位关键输出

```bash
# 对比图
!ls -la outputs_batch_groups/compare/ 2>/dev/null || ls -la outputs_by_scene/*/compare/ 2>/dev/null | head -20

# 分组叠底图
!find outputs_batch_groups/groups -type f -name "heatmap.png" 2>/dev/null | head -n 10

# 投稿组合图
!ls -la figures/Fig_heatmap_6scenes.png 2>/dev/null
```

你重点看：
- `outputs_batch_groups/groups/Overall/heatmap.png`
- `outputs_batch_groups/groups/Experience-High/heatmap.png`
- `outputs_batch_groups/groups/Experience-Low/heatmap.png`
- `outputs_batch_groups/compare/Experience_comparison.png`
- `figures/Fig_heatmap_6scenes.png`（6 场景组合投稿图）

---

## Cell 9：打包并下载

```bash
!zip -qr outputs_batch_groups.zip outputs_batch_groups figures/
```

```python
from google.colab import files
files.download('outputs_batch_groups.zip')
```

---

## 常见坑（很重要）

1) **叠底是否对齐**只取决于：底图像素尺寸 == `screen_w/screen_h`，且 gaze 坐标与底图同一 px 坐标系。

2) CSV 至少要有列名（或能映射到）：
- `Gaze Point X[px]`
- `Gaze Point Y[px]`
- `Fixation Point X[px]` / `Fixation Point Y[px]`（默认 fixation 模式需要）
- `Fixation Duration[ms]`（默认 duration 加权需要）

3) 你们现在是「每人 1 个 CSV」，并且文件名含 name；脚本会按 name 匹配 CSV，如果一个 name 匹配到多个 CSV 会报错（防止混淆）。

4) manifest 只需要 `name` + `Experience` 列。`SportFreq` 列如果存在也不影响，但不再被热力图脚本使用。
