# eyetrack Colab 逐 Cell 复制版（最新）

> 目的：你不想打开现成 Notebook，而是希望把每段代码当作一个 Colab cell 逐个复制粘贴运行。
>
> 本文覆盖：
> - 批处理热图（不做 AOI）
> - 分组批处理（SportFreq 二分 / Experience 二分 / 4 类交叉）
> - 差异图（log2 ratio）
> - 热图叠加统一底图（overlay）

---

## 你需要提前准备并上传到 Colab 的文件

- `csvs.zip`：眼动 CSV 数据
  - 旧版结构：所有 CSV 都在同一个文件夹（**每人 1 个 CSV**）
  - **新版结构（推荐/常见）**：按场景分文件夹（例如 12 个场景文件夹）；每个场景文件夹内包含该场景下所有人的 CSV（每人 1 个）
- `group_manifest.csv`：表头为 `name,SportFreq,Experience`（每人 1 行）
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

## Cell 3：上传文件（csvs.zip / group_manifest.csv / scene.png）

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

### 4B) 新版结构（推荐）：解压为“按场景分文件夹” scenes/

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

## Cell 6：运行（分组批处理 + 差异图 + 叠底图）

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
  --screen_w 1920 --screen_h 1080 \
  --outdir outputs_by_scene
```

输出结构示例：
- `outputs_by_scene/<scene_name>/individual/...`
- `outputs_by_scene/<scene_name>/groups/...`
- `outputs_by_scene/<scene_name>/compare/...`

---

## Cell 7：快速定位关键输出

```bash
# 差异图
!ls -la outputs_batch_groups/compare

# 分组叠底图（示例列出前 30 张）
!find outputs_batch_groups/groups -type f -name "heatmap_overlay.png" | head -n 30
```

你重点看：
- `outputs_batch_groups/compare/SportFreq_diff.png`（叠底）
- `outputs_batch_groups/compare/Experience_diff.png`（叠底）
- `outputs_batch_groups/compare/4way_grid.png`（叠底）
- `outputs_batch_groups/groups/**/heatmap.png`（叠底）
- `outputs_batch_groups/individual/<name>/heatmap.png`（叠底）

---

## Cell 8：打包并下载

```bash
!zip -qr outputs_batch_groups.zip outputs_batch_groups
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

3) 你们现在是「每人 1 个 CSV」，并且文件名含 name；脚本会按 name 匹配 CSV，如果一个 name 匹配到多个 CSV 会报错（防止混淆）。
