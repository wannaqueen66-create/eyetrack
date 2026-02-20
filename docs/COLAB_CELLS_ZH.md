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

- `csvs.zip`：包含所有人的 CSV（**每人 1 个 CSV**；文件名包含 name 用于匹配）
- `group_manifest.csv`：表头为 `name,SportFreq,Experience`
- `scene.png`（或 jpg）：所有人同一场景的一张底图（像素尺寸要与 screen_w/screen_h 一致）

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

---

## Cell 3：上传文件（csvs.zip / group_manifest.csv / scene.png）

```python
from google.colab import files
files.upload()
```

---

## Cell 4：解压 CSV 到同一文件夹 batch_csvs/

```bash
!rm -rf batch_csvs
!mkdir -p batch_csvs
!unzip -q csvs.zip -d batch_csvs

# quick check
!find batch_csvs -maxdepth 2 -type f -name "*.csv" | head
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

```bash
!python scripts/batch_heatmap_groups.py \
  --manifest group_manifest.csv \
  --csv_dir batch_csvs \
  --screen_w 1920 --screen_h 1080 \
  --background_img scene.png \
  --outdir outputs_batch_groups
```

---

## Cell 7：快速定位关键输出

```bash
# 差异图
!ls -la outputs_batch_groups/compare

# 分组叠底图（示例列出前 30 张）
!find outputs_batch_groups/groups -type f -name "heatmap_overlay.png" | head -n 30
```

你重点看：
- `outputs_batch_groups/compare/SportFreq_diff.png`
- `outputs_batch_groups/compare/Experience_diff.png`
- `outputs_batch_groups/compare/4way_grid.png`
- `outputs_batch_groups/groups/**/heatmap_overlay.png`

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
