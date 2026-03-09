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

主线 PNG 风格说明：
- `main` 上的核心 PNG 已统一为更接近 Origin / Building and Environment 期刊气质的清爽投稿风格。
- 影响范围包括：描述性 grouped PNG、解释型 LMM PNG、evidence PNG，以及 fixed-effect forest plot。
- 重点优化了配色、留白、网格线、标题/图例层级，以及数值标签的避免重叠策略。
- 为了兼顾可读性与结果保真，主线 PNG 现在更系统地配套 `*_data.csv`，包括描述性 grouped 图、解释型 LMM 图、evidence 图和 forest plot；当图内不再铺满所有数值标签时，仍可直接对照精确结果。

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

### 每套 `02_显著性分析_Significance` 内部建议阅读顺序
全样本与 QC 后两套结果，建议都按同一顺序读：

1. `allocation_lmm/groupvar_Experience/tables/model_family_index.csv`  
   先看三套 LMM 家族的索引：main effects / two-way interactions / three-way interaction。
2. `allocation_lmm/groupvar_Experience/tables/three_model_packet_summary.csv`  
   这是最快的一张三模型总览表，用来判断哪些 outcome × family 足够稳定，可以进入主结果。
3. 然后按 `01_main_effects` → `02_two_way_interactions` → `03_three_way_interaction` 的顺序进入各模型家族文件夹。
4. 先只看主显著性指标：`share_pct`、`share_logit`、`FC_share`、`fc_share_logit`、`FC_rate`、`tfd_y`、`ttff_y`、`fc_y`。
5. 对每个主指标，在对应模型家族内再按一个固定包读取：
   - `model_stability_summary.csv` / `evidence_stability_overview_Experience.png`
   - `model_fit_<outcome>.csv` → 模型是否可用
   - `fixef_<outcome>.csv` → 该模型家族下的 fixed effects 主表
   - `wwr_trend_tests_<outcome>.csv` / `wwr_trend_coding_<outcome>.csv` → 明确回答 WWR 是线性上升/下降，还是中点（通常 45）最高/最低
   - `contrasts_<outcome>.csv` → 面向 simple effects / 审稿回复的 contrasts，主要看交互模型家族
   - `evidence_model_fit_overview_Experience.png` / `evidence_fixef_key_terms_<outcome>.png` / `evidence_wwr_trend_terms_<outcome>.png` / `evidence_wwr_trend_shape_<outcome>.png` / `evidence_contrasts_<outcome>.png` → 面向论文和审稿回复的证据图
6. `allocation_lmm_visuals/`  
   作为解释图、辅图来读，不要替代核心统计表。
7. `two_part_models/`  
   当 scene feature 问题是主问题时再并入主线；否则可放补充。
8. `ffd_y`、`mfd_y`、`rff_y`、`MPD` 这些探索性指标，优先放补充或机制讨论。

### `overall` 与 `Experience` 在显著性主线里的关系
- `overall` 继续保留在**描述性主线**里，适合交代水平、分布、整体形态。
- 当前 **显著性主线** 主要围绕 `allocation_lmm/` 中的 `Experience`（其次是 `SportFreq`）展开。
- 也就是说：`grouped_overall` 不应直接和显著性主结果混成一条证据链；overall 负责背景，Experience LMM 负责主统计结论。

---

## main 主线目前可见的 AOI 指标

当前 main 主线的底表与描述性输出，已经可以在源眼动导出列足够时暴露以下 AOI 指标：

- **FC** = Fixation Count（注视次数）
- **FFD** = First Fixation Duration（首次注视时长）
- **MFD** = Mean Fixation Duration（平均注视时长）
- **MPD** = Mean Pupil Diameter（平均瞳孔直径）
- **RFF** = Re-fixation Frequency（重返/重新进入频次）
- **TFD** = Total Fixation Duration（总注视时长）
- **TTFF** = Time To First Fixation（首次进入时延）
- **FC_share / FC_prop / FC_rate** = FC 的归一化伴随指标
- **share / share_pct** = TFD 的归一化伴随指标

### 哪些适合主显著性主线，哪些更适合描述/探索

**Tier 1：主结果显著性指标（当前 `main` 上推荐的 headline 顺序）**
- `share_pct` / `share_logit`（基于 TFD 的注意分配占比）
- `FC_share` / `fc_share_logit`（基于 FC 的注意分配占比）
- `FC_rate`
- `TFD`
- `TTFF`
- `FC`

这些指标最贴合当前主线里“分配 / 时延 / 绝对注意量”三类问题。若你想给论文正文和审稿回复建立一条稳定、可复用的主结果梯队，优先用这一层。

**Tier 2：补充 / 探索 / 机制支持指标**
- `FFD`
- `MFD`
- `RFF`
- `MPD`

它们已经进入底表、描述性输出，也接入了探索性的 LMM 风格输出；但在当前 `main` 上，更适合作为支持性或探索性结果，而不是第一主结论。

### 读 LMM 文件时建议对照的 outcome 命名
显著性目录里的部分文件名会使用模型内部 outcome 名：

- `share_pct` = TFD 注意分配占比（百分比形式）
- `share_logit` = logit 变换后的 TFD 注意分配占比
- `FC_share` = FC 注意分配占比（原比例）
- `fc_share_logit` = logit 变换后的 FC 注意分配占比
- `FC_rate` = 每秒 FC 速率
- `tfd_y` = `log1p(TFD)`
- `ttff_y` = `visited==1` 条件下的 `log1p(TTFF)`
- `fc_y` = `visited==1` 条件下的 `log1p(FC)`
- `ffd_y` = `visited==1` 条件下的 `log1p(FFD)`，探索性
- `mfd_y` = `visited==1` 条件下的 `log1p(MFD)`，探索性
- `rff_y` = `log1p(RFF)`，探索性
- `MPD` = 原始平均瞳孔直径，探索性

### 指标定义与处理口径

#### FC
原始 AOI fixation count 保留为 `FC`。
**不要**再手工按组人数去“归一化”。
需要标准化时优先看：
- `FC_share` / `FC_prop`：trial 内次数占比
- `FC_rate`：按 trial 时长折算的每秒速率

#### TFD
原始 AOI 总注视时长保留为 `TFD`。
如果你关注“注意分配”，继续优先看：
- `share`
- `share_pct`

#### TTFF
`TTFF` 延续现有 segment-aware QC 逻辑。
如果检测到 video time reset 或大 gap，会按 segment 重新计算 TTFF，并通过 `ttff_source`、`ttff_warning`、`ttff_qc_status` 等列标记上下文。

#### FFD
这里定义为：**首次进入 AOI 的那个唯一 fixation 的持续时间**。
推荐用途：早期吸引/首次进入后的停留深度。
注意：应只在 `visited==1` 的前提下解释。

#### MFD
这里定义为：**AOI 内所有唯一 fixation 的平均持续时间**。
推荐用途：持续加工/停留深度的描述性指标。
注意：它会受 fixation parser/export 规则影响，建议主要作为描述性或探索性支撑。

#### RFF
这里定义为：**基于 fixation 序列，首次进入后再次重新进入 AOI 的 episode 次数**。
推荐用途：回看/重返倾向。
注意：它不是“每分钟重返率”，而是序列派生的 revisit 指标；最好结合 FC / share 一起看。

#### MPD
这里定义为：**AOI 子集上的平均瞳孔直径**，优先平均左右眼 mm 列；若 mm 列缺失，则回退到 px 列。
推荐用途：生理负荷或状态的描述性背景信息。
注意：MPD 对设备、导出格式、光照条件都很敏感，除非采集条件控制得很好，否则建议保持探索性解释。

### 重跑后去哪里看这些指标

执行 `python scripts/run_analysis2.py ...` 后，最容易在下面这些位置看到新增指标：

**底表 / 原始 AOI 输出**
- `研究输出_时间戳/*/98_附录_Appendix/raw_batch_outputs/batch_aoi_metrics_by_class.csv`
- `研究输出_时间戳/*/98_附录_Appendix/raw_batch_outputs/batch_aoi_metrics_by_polygon.csv`

**描述性输出**
- `研究输出_时间戳/*/01_描述性分析_Descriptive/grouped_overall/tables/summary_Experience_*.csv`
- `研究输出_时间戳/*/01_描述性分析_Descriptive/grouped_overall/tables/summary_Experience_*.csv`
- `研究输出_时间戳/*/01_描述性分析_Descriptive/grouped_overall/png/plot_Experience_*.png`
- `研究输出_时间戳/*/01_描述性分析_Descriptive/grouped_overall/data/plot_Experience_*_data.csv`
- `研究输出_时间戳/*/01_描述性分析_Descriptive/grouped_experience/tables/`  
  这里是 Experience 描述性主线的便捷镜像，按表 / 图 / data 分开。
- `研究输出_时间戳/*/01_描述性分析_Descriptive/grouped_experience/png/`
- `研究输出_时间戳/*/01_描述性分析_Descriptive/grouped_experience/data/`
- `研究输出_时间戳/*/01_描述性分析_Descriptive/organized_outputs/grouped/tables/summary_experience_scene.csv`
- `研究输出_时间戳/*/01_描述性分析_Descriptive/organized_outputs/grouped/png/*.png`
- `研究输出_时间戳/*/01_描述性分析_Descriptive/organized_outputs/grouped/data/*_data.csv`

**显著性 / 探索性推断输出**
- `研究输出_时间戳/*/02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/`  
  这里先看主线索引表与三模型总览表。
- `研究输出_时间戳/*/02_显著性分析_Significance/allocation_lmm/groupvar_Experience/01_main_effects/{tables,png,data}/`
- `研究输出_时间戳/*/02_显著性分析_Significance/allocation_lmm/groupvar_Experience/02_two_way_interactions/{tables,png,data}/`
- `研究输出_时间戳/*/02_显著性分析_Significance/allocation_lmm/groupvar_Experience/03_three_way_interaction/{tables,png,data}/`
- `研究输出_时间戳/*/02_显著性分析_Significance/allocation_lmm_visuals/tables/`
- `研究输出_时间戳/*/02_显著性分析_Significance/allocation_lmm_visuals/png/`
- `研究输出_时间戳/*/02_显著性分析_Significance/allocation_lmm_visuals/data/`

标准命名与定义也可参考：`docs/METRICS_SPEC.md`。
显著性主线的固定阅读顺序可参考：`docs/SIGNIFICANCE_MAINLINE.md`。

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
