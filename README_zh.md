# eyetrack（主线精简版说明）

这个仓库的 `main` 分支现在只强调两大块输出主线：

1. `01_描述性分析_Descriptive`
2. `02_显著性分析_Significance`

此前更复杂、更完整的历史状态已经保存在 **`raw` 分支**。
如果你需要旧版 README、旧命名、旧目录层次，直接切到 `raw` 看。

---

## main 分支现在在讲什么

### 一、01_描述性分析_Descriptive
这是现在的**描述性主线**。

当前在 `main` 上优先突出：
- overall（整体整理视图）
- Experience 分组

当前被降级、不再作为首页主线强调的内容：
- 旧的 analysis2 / research_bundle 叙事
- 过多并列的历史目录命名
- 对当前论文主叙事帮助不大的扩展展示层

### 二、02_显著性分析_Significance
这是现在的**显著性分析主线**。

主内容包括：
- allocation LMM
- 面向解释的 LMM 可视化
- two-part models（当 scene features 可用时）

### 三、附录 / 历史保留内容
这些内容没有删除，但不再放在主舞台：
- diagnostics
- raw AOI batch outputs
- Colab 说明与实现说明
- 旧命名兼容入口脚本

---

## 分支说明

### `main`
用于当前更清爽、论文导向的主线版本。

### `raw`
用于保留整理前的复杂现状。
适合在以下情况查看：
- 想找旧版 README 结构
- 想看旧输出命名方式
- 想保留完整历史语境
- 想直接对照整理前版本

---

## 当前推荐命令

### 本地 / VPS

```bash
cd /root/.openclaw/workspace/eyetrack
source .venv/bin/activate
python scripts/run_analysis2.py \
  --group_manifest /path/to/group_manifest.csv \
  --scenes_root /path/to/scenes_root
```

### Colab

```bash
cd /content/eyetrack
python scripts/run_colab_one_command.py \
  --scenes_root_orig /content/drive/MyDrive/映射 \
  --group_manifest /content/drive/MyDrive/映射/group_manifest.csv
```

---

## main 分支下的输出结构

```text
研究输出_YYYYMMDD_HHMMSS/
├─ 01_描述性分析_Descriptive/
│  ├─ organized_outputs/
│  ├─ grouped_overall/
│  └─ grouped_experience/
├─ 02_显著性分析_Significance/
│  ├─ allocation_lmm/
│  ├─ allocation_lmm_visuals/
│  └─ two_part_models/
└─ 98_附录_Appendix/
   ├─ diagnostics/
   ├─ raw_batch_outputs/
   ├─ notes/
   └─ colab_notes/
```

---

## 现在什么是主线，什么被降级了

### 主线内容
- `01_描述性分析_Descriptive/organized_outputs`
- `01_描述性分析_Descriptive/grouped_overall`
- `01_描述性分析_Descriptive/grouped_experience`
- `02_显著性分析_Significance/allocation_lmm`
- `02_显著性分析_Significance/allocation_lmm_visuals`
- `02_显著性分析_Significance/two_part_models`

### 已降级但保留
- “research bundle” 这套旧说法
- “analysis2” 作为对外主叙事
- 原始 AOI 批处理导出
- 诊断信息和实现说明
- 老的兼容别名脚本

原则是不删有用内容，只把它们从主线移开：
- 要么放到附录/兼容层
- 要么保留在 `raw` 分支

---

## 最小环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 现在优先看的入口脚本

- `scripts/run_analysis2.py`：当前主线总入口
- `scripts/run_colab_one_command.py`：当前 Colab 主入口
- `scripts/run_aoi_metrics.py`：单次 AOI 指标计算
- `scripts/run_minimal_aoi_bundle.py`：最小四输入入口

仓库里仍保留一些旧别名脚本，但它们不再是 `main` 分支想强调的主路径。

---

## 兼容与历史说明

仓库中仍可能看到这些旧词：
- `research_bundle`
- `analysis2`
- `03_TwoPart模型`
- `00_AOI原始批处理`

这些名称现在主要用于：
- 兼容旧脚本
- 保持历史连续性
- 方便从 `raw` 分支迁移

如果你只想抓当前主线，请直接按“描述性分析 + 显著性分析”这两大块理解整个仓库。
