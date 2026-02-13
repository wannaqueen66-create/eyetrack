# eyetrack Colab 部署与使用流程（中文）

## 一键入口

- 标准版（中英混合）：
  https://colab.research.google.com/github/wannaqueen66-create/eyetrack/blob/main/notebooks/eyetrack_colab_quickstart.ipynb
- 纯中文逐 Cell 说明版：
  https://colab.research.google.com/github/wannaqueen66-create/eyetrack/blob/main/notebooks/eyetrack_colab_quickstart_zh.ipynb

---

## 使用步骤

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

