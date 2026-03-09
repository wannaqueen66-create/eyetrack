# eyetrack（主线精简版说明）

这个仓库的 `main` 分支现在明确输出 **两套结果轨道**，并且每套内部继续只强调 **两大分析块**：

1. `00_全样本_AllSample`
2. `01_QC后_AfterQC`

每套内部包含：
- `01_描述性分析_Descriptive`
- `02_显著性分析_Significance`

此前更复杂、更完整的历史状态已经保存在 **`raw` 分支**。
如果你需要旧版 README、旧命名、旧目录层次，直接切到 `raw` 看。

---

## main 分支现在在讲什么

### 一、结果轨道 1：`00_全样本_AllSample`
这是**全样本结果**，基于完整被试清单跑出的主线输出。

### 二、结果轨道 2：`01_QC后_AfterQC`
这是 **QC 后结果**，会先排除以下 8 个被试，再基于过滤后的 participant manifest **重新整套分析**：

- 孙校聪
- 康少勇
- 张钰鹏
- 杨可
- 洪婷婷
- 陈韬
- 高梓楠
- 赵国宏

注意：
- 这不是最后画图时临时把人筛掉；
- 而是主线 orchestrator 先生成过滤后的 manifest，再把整套描述性分析 / 显著性分析重新跑一遍；
- 排除名单记录在 `configs/excluded_participants_qc.csv`；
- 说明文档见 `docs/QC_EXCLUSIONS.md`。

### 三、每套结果内部的两大分析块

#### `01_描述性分析_Descriptive`
这是现在的**描述性主线**。

当前在 `main` 上优先突出：
- `organized_outputs`
- `grouped_overall`
- `grouped_experience`

#### `02_显著性分析_Significance`
这是现在的**显著性分析主线**。

主内容包括：
- allocation LMM
- 面向解释的 LMM 可视化
- two-part models（当 scene features 可用时）

### 四、附录 / 历史保留内容
这些内容没有删除，但不再放在主舞台：
- diagnostics
- raw AOI batch outputs
- Colab 说明与实现说明
- 旧命名兼容入口脚本

---

## 分支说明

### `main`
用于当前更清爽、论文导向的主线版本，并且显式分成：
- 全样本
- QC后

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

正常使用下，命令**不需要新增必填参数**。QC 后重跑已经内置到主线 orchestrator 中。
如果你确实要换一份 QC 排除名单，可以额外传：

```bash
--qc_exclusion_csv /path/to/excluded_participants_qc.csv
```

---

## main 分支下的新输出结构

```text
研究输出_YYYYMMDD_HHMMSS/
├─ 00_全样本_AllSample/
│  ├─ 01_描述性分析_Descriptive/
│  │  ├─ organized_outputs/
│  │  ├─ grouped_overall/
│  │  └─ grouped_experience/
│  ├─ 02_显著性分析_Significance/
│  │  ├─ allocation_lmm/
│  │  ├─ allocation_lmm_visuals/
│  │  └─ two_part_models/
│  └─ 98_附录_Appendix/
│     ├─ diagnostics/
│     ├─ raw_batch_outputs/
│     ├─ notes/
│     └─ colab_notes/
├─ 01_QC后_AfterQC/
│  ├─ 01_描述性分析_Descriptive/
│  ├─ 02_显著性分析_Significance/
│  └─ 98_附录_Appendix/
└─ README_研究输出说明.txt
```

---

## 跑完后应该先看哪里

### 看全样本结果
先看：
- `研究输出_时间戳/00_全样本_AllSample/01_描述性分析_Descriptive`
- `研究输出_时间戳/00_全样本_AllSample/02_显著性分析_Significance`

### 看 QC 后结果
先看：
- `研究输出_时间戳/01_QC后_AfterQC/01_描述性分析_Descriptive`
- `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance`

---

## 最小环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 现在优先看的入口脚本

- `scripts/run_analysis2.py`：当前主线总入口（现在会同时产出“全样本 + QC后”两套结果）
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

如果你只想抓当前主线，请直接按“全样本 + QC后”两套结果、且每套内部再看“描述性分析 + 显著性分析”这两大块理解整个仓库。
