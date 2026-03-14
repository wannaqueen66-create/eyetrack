# 结果章节草稿（中文，带图号/表号版）

> 适用结果包：clean-main 双轨输出（`00_全样本_AllSample` / `01_QC后_AfterQC`）
>
> 推荐正文口径：以 **QC后轨道** 为主，全样本轨道作为稳健性对照。
>
> 当前版本定位：论文结果章节第一版草稿，强调结构、叙述顺序与图表落位；后续可再根据最终投稿目标与统计口径细化措辞。

---

## 4 结果

### 4.1 样本与 QC 口径

本研究的眼动结果采用 clean-main 双轨输出结构，分别为 **全样本轨道（All Sample）** 与 **QC 后轨道（After QC）**。正文结果优先报告 **QC 后轨道**，以减少异常样本对核心结论的干扰；全样本轨道用于补充说明结果方向的一致性与稳健性。

本次实验共包含 12 个正式场景，每个场景均包含背景图、AOI 标注文件以及多名被试的原始眼动 CSV。AOI 主要分为 3 类：**table**、**window** 与 **equipment**。场景条件由 **WWR（15/45/75）** 与 **Complexity（C0/C1）** 共同构成，并在两个 round 中重复呈现。

在 QC 后轨道中，共排除 8 名被试：孙校聪、康少勇、张钰鹏、杨可、洪婷婷、陈韬、高梓楠、赵国宏。QC 后轨道采用过滤 participant manifest 后的整套重跑，而非仅在作图阶段隐藏相应个体，因此其结果可视为严格意义上的 QC 后分析结果。

从分组规模看，QC 后轨道中 Experience 分组分别包含 **High 组 15 人、Low 组 7 人**；全样本轨道中 Experience 分组分别包含 **High 组 19 人、Low 组 9 人**。因此，后续正文在比较组间差异时，需同时考虑样本量不平衡与模型稳定性问题。

> **建议插入表格**
>
> - **表4-1** 样本与 QC 轨道概览（可根据 `group_size_summary_Experience.csv` 整理）
>
> **建议说明来源**
>
> - QC：`01_QC后_AfterQC/README_研究输出说明.txt`
> - QC exclusions：`configs/excluded_participants_qc.csv`

---

### 4.2 Experience 分组下的描述性模式

在正式进入推断模型结果前，首先从描述性图形概览 Experience 分组在不同 AOI、WWR 与 Complexity 条件下的注意分配模式，以建立对整体数据结构的直观认识。

#### 4.2.1 注意分配占比（share\_pct）的总体模式

图4-1 展示了 QC 后样本中，按 Experience 分组的 AOI 注意分配占比（`share_pct`，即 TFD 在 trial 内的百分比）随 WWR 与 Complexity 的变化模式。整体上，**table AOI 的分配占比最高**，其次为 **equipment**，而 **window** 的分配占比最低。这一层级关系在高经验组与低经验组中均可观察到，说明 AOI 类型本身对注意分配具有稳定影响。

进一步看条件层面，`table` AOI 在 **C0 条件** 下通常获得更高的注意分配，而在 **C1 条件** 下占比明显下降。例如，在 round 1 中，`table` 的 share\_pct 在 WWR15-C0 与 WWR45-C0 条件下均维持在较高水平，而在 WWR15-C1 与 WWR45-C1 条件下降低。该趋势在 round 2 中仍然可以观察到，提示复杂度变化可能改变被试对关键物体区域的注意集中程度。

Experience 组间差异总体存在，但其表现更像是“在特定 AOI × 条件组合中的幅度差异”，而不是一个稳定、单调的整体主效应。也就是说，经验水平更可能通过调节特定条件下的 AOI 分配模式来影响结果，而不是简单地导致所有场景中注意分配整体上升或下降。

> **建议正文主图**
>
> - **图4-1** Experience 分组下 AOI 注意分配占比（share\_pct）随 WWR × Complexity 的变化图  
>   文件建议：`01_QC后_AfterQC/01_描述性分析_Descriptive/grouped_experience/png/plot_Experience_share_pct.png`

#### 4.2.2 注视次数与注视率的补充模式

除时长型分配指标外，图4-2 可进一步展示基于注视次数或注视率的补充模式（建议在 `FC_share` 与 `FC_rate` 中择一放入正文）。这些图的作用主要在于说明：被试在不同 AOI 上的注意差异，不仅体现在停留时长上，也可能体现在注视进入频率或注视次数分配上。

如果正文更强调“注意资源在不同 AOI 之间的相对分配”，则更推荐使用 `FC_share`；如果更强调“注视行为节奏与密度”，则可优先考虑 `FC_rate`。从当前图形可视化质量与解释直接性看，`FC_share` 更适合与 `share_pct` 形成配套叙述。

> **建议正文次主图（二选一）**
>
> - **图4-2a** Experience 分组下 FC\_share 图  
>   文件建议：`plot_Experience_FC_share.png`
>
> 或
>
> - **图4-2b** Experience 分组下 FC\_rate 图  
>   文件建议：`plot_Experience_FC_rate.png`

#### 4.2.3 注意进入时延（TTFF）的描述性特征

图4-3 展示了 Experience 分组下 TTFF 的条件变化模式。与 `share_pct` 相比，TTFF 的组间差异更不稳定，且受个体差异与场景条件波动影响更大，表现为置信区间较宽、局部趋势不够平滑。因此，TTFF 更适合作为“注意进入机制”的补充指标，而不宜单独承担正文主结论。

从当前结果看，TTFF 的变化方向在不同 AOI 上并不完全一致：某些条件组合下，window 的首次进入时延在低经验组中偏高；而在另一些组合中，经验组差异并不显著。这提示 TTFF 更可能反映场景导向与进入策略，而不是简单的资源分配总量。

> **建议正文补充图 / 或补充材料首图**
>
> - **图4-3** Experience 分组下 TTFF 的 WWR × Complexity 描述性图  
>   文件建议：`plot_Experience_TTFF.png`

---

### 4.3 Experience 主线的推断结果

#### 4.3.1 报告原则：先看模型稳定性，再解释效应

本研究的推断性分析以 Experience 为主分组变量，采用 allocation LMM 结果包，按以下三套模型家族组织：

1. 主效应（Main effects）
2. 两因素交互（Two-way interactions）
3. 三因素交互（Three-way interaction）

考虑到当前数据在多个 outcome 上存在模型收敛、随机效应奇异或标准误异常等问题，正文对推断结果采用“**稳定性优先**”的报告原则，即：

- 先检查 `model_family_index.csv` 与 `three_model_packet_summary.csv` 的稳定性等级；
- 仅对达到 **stable** 或 **caution** 的 outcome × family 组合进行主文解释；
- 对明显 **unstable** 或 `fit_failed` 的结果，保留为探索性材料，不作为主结论依据。

> **建议正文表格**
>
> - **表4-2** Experience 主线推断包的稳定性概览  
>   文件建议：`three_model_packet_summary.csv`

#### 4.3.2 QC 后轨道：以描述性与解释性图为主，谨慎使用推断结果

在 QC 后轨道中，大部分主 outcome（如 `share_pct`、`FC_share`、`tfd_y`、`ttff_y` 等）在主效应或交互模型中都存在不同程度的不稳定问题。例如，部分模型出现 Hessian 非正定、随机效应边界奇异、标准误异常，或在更高阶交互中直接拟合失败。基于这一点，QC 后轨道中的推断结果更适合作为“趋势支持”与“解释补充”，而不宜作为单独的硬性统计主结论。

因此，正文建议采取以下叙述方式：

- 以 **图4-1** 所示的描述性模式为主线；
- 再用 **图4-4**（解释性条件交互图）补充说明 Experience 在不同条件下的方向差异；
- 在文字中明确交代：QC 后 LMM 在多个 outcome 上稳定性不足，因此正文主要依据描述性结构与更稳定的对照结果来组织解释。

> **建议正文主图（解释性图）**
>
> - **图4-4** Experience 组别在不同场景条件下对 AOI 注意分配占比的解释性条件交互图  
>   文件建议：`01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm_visuals/png/condition_group_interaction_Experience_share_pct.png`

#### 4.3.3 全样本轨道：作为稳健性对照的稳定结果

在全样本轨道中，主 outcome 中仅少数结果达到可稳定解释的水平。当前较为稳健的结果是 `share_logit` 在主效应模型中的表现。该模型的 `model_fit_share_logit.csv` 显示其稳定等级为 **stable**，说明在当前数据结构下，`share_logit` 比原始比例型指标更适合进入主效应推断链条。

在该稳定模型中，WWR 主效应为负，提示随着 WWR 增大，AOI 注意分配的 logit 形式总体发生下降；同时，window 相对于基准 AOI 的 WWR 交互项显著为正，说明 WWR 的变化对不同 AOI 的影响并不一致。换言之，条件变化并不是均匀作用于所有 AOI，而是更可能重塑被试在不同视觉区域之间的分配结构。

这一结果为正文提供了一个重要的稳健性支撑：虽然 QC 后轨道的模型在多个 outcome 上不够稳定，但全样本轨道中至少有一个核心 outcome（`share_logit`）在主效应层面提供了可解释的统计证据，方向上与描述性图所揭示的 AOI 分配结构变化是一致的。

> **建议正文表格**
>
> - **表4-3** 全样本轨道中 `share_logit` 主效应模型的拟合与固定效应摘要  
>   文件建议：
>   - `model_fit_share_logit.csv`
>   - `fixef_share_logit.csv`

---

### 4.4 QC 后与全样本结果的对照解释

综合两条轨道，可以形成如下更稳健的结果判断：

1. **描述性结构高度一致。** 无论在 QC 后轨道还是全样本轨道中，AOI 分配都表现出明显的类型层级，即 `table` 通常获得更高的注意资源，而 `window` 与 `equipment` 相对较低。

2. **条件效应更像“结构重分配”，而非简单强弱变化。** WWR 与 Complexity 的变化并不是让所有 AOI 的指标同向变化，而更像是在不同 AOI 之间重新分配注意比例。

3. **Experience 的作用存在，但不是一个孤立主效应。** Experience 更可能通过调节“在哪些条件下、对哪些 AOI 的分配更强/更弱”来体现，而不是在所有指标上表现为单一方向的整体差异。

4. **模型稳定性限制了高阶交互的强解释。** 因此，正文更适合以描述性图和解释性图构建主要叙事，再用全样本中的稳定模型作为统计支撑，而不是反过来让复杂模型主导全文。

---

### 4.5 图表落位建议（正文 / 补充材料）

#### 正文优先图

- **图4-1** `plot_Experience_share_pct.png`
- **图4-2** `plot_Experience_FC_share.png` 或 `plot_Experience_FC_rate.png`（二选一）
- **图4-3** `plot_Experience_TTFF.png`（若篇幅允许）
- **图4-4** `condition_group_interaction_Experience_share_pct.png`

#### 补充材料优先图

- `scene_group_profile_Experience_share_pct.png`
- 全部 SportFreq explanatory visuals
- FFD / MFD / RFF / MPD 系列图
- 大多数 scene-level profile 图

#### 正文核心表

- **表4-1** 样本与 QC 轨道概览
- **表4-2** Experience 主线 LMM 稳定性总览
- **表4-3** 全样本 `share_logit` 主效应模型结果摘要

---

### 4.6 小结

总体而言，本研究的眼动结果显示，不同 AOI 之间存在稳定的注意分配层级结构，且该结构会随着场景条件（WWR 与 Complexity）的变化而发生系统性调整。Experience 的作用更多体现为在特定条件与特定 AOI 上的差异化调节，而非单一的整体主效应。尽管 QC 后轨道中的高阶推断模型存在稳定性不足的问题，但描述性结果与全样本轨道中的稳定模型共同表明：场景条件确实会改变被试在不同 AOI 之间的注意分配结构，而这一过程受到经验水平的影响。

---

## 附：建议配图方案（最稳妥版）

### 方案 A（推荐）
1. 图4-1：`plot_Experience_share_pct.png`
2. 图4-2：`plot_Experience_FC_share.png`
3. 图4-3：`plot_Experience_TTFF.png`
4. 图4-4：`condition_group_interaction_Experience_share_pct.png`

### 方案 B（更精简）
1. 图4-1：`plot_Experience_share_pct.png`
2. 图4-2：`condition_group_interaction_Experience_share_pct.png`
3. 图4-3：`plot_Experience_FC_share.png`

---

## 使用提醒

- 如果你后面想把结果章写得更“像投稿稿”，建议下一步把本文件进一步压缩成：
  - **短版正文段落**
  - **对应图注草稿**
  - **表格引用句式**
- 如果你希望结果段落更保守，可把“显著”“作用”这类词适当替换成“提示”“表明”“支持了……的趋势”。
