# 结果阅读指南（eyetrack main 主入口）

这份文件给老师、合作者、第一次接触当前眼动 `main` 分支结果包的人用。

如果你已经拿到了一个 `研究输出_时间戳/` 文件夹，但不知道先看什么，就从这里开始。

---

## 一句话先说清
当前 `main` 分支把结果明确组织成两条轨道：

1. **00_全样本_AllSample**
2. **01_QC后_AfterQC**

并且每条轨道内部只强调两个主块：
- **描述性分析**
- **显著性分析**

如果是正式汇报、写论文、给导师发结果，优先看 **QC后轨道**。

---

## 第一步：先看总说明
打开结果包后，先读：
1. `研究输出_时间戳/README_研究输出说明.txt`
2. `研究输出_时间戳/01_QC后_AfterQC/README_研究输出说明.txt`

原因：
- 先知道双轨结构是什么意思
- 先知道 QC 后结果不是“图上隐藏”，而是基于过滤后 manifest 的整套重跑
- 先建立“描述性 / 显著性”两大块的地图感

---

## 第二步：先看描述性结果

### 2.1 先看全样本背景
先看：
- `研究输出_时间戳/00_全样本_AllSample/01_描述性分析_Descriptive/grouped_overall/png/`

再看：
- `.../grouped_overall/data/`
- `.../grouped_overall/tables/`

你会得到：
- 总体水平和整体模式
- 分布大致稳不稳
- 哪些 outcome 值得后面重点看显著性主线

### 2.2 再看 Experience 分组描述性结果
这通常比 overall 描述性更接近当前 main 的主结论路径。

先看：
- `研究输出_时间戳/01_QC后_AfterQC/01_描述性分析_Descriptive/grouped_experience/png/`

再看：
- `.../grouped_experience/data/`
- `.../grouped_experience/tables/`

你会得到：
- Experience 高低组的直观差异
- 哪些 outcome 可能进入主显著性链条

一句话规则：
- **先看 PNG 抓模式**
- **再看 CSV / data 对数值**

---

## 第三步：看显著性主线

### 3.1 先看 Experience LMM 总索引
最高优先级：
- `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/model_family_index.csv`
- `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/three_model_packet_summary.csv`

这两张表负责告诉你：
- 当前 LMM 结果分成哪三类模型家族
- 哪些 outcome × family 的稳定性足够高，可以当主结果候选

### 3.2 再按三套模型家族顺序进入
固定顺序：
1. `01_main_effects`
2. `02_two_way_interactions`
3. `03_three_way_interaction`

这样读的好处是：
- 不会一上来就跳进最复杂的三阶交互
- 能让统计叙事更稳定

---

## 第四步：每个 outcome 固定按一个 packet 读
在每个模型家族内部，建议按下面顺序：
1. `model_stability_summary.csv`
2. `model_fit_<outcome>.csv`
3. `fixef_<outcome>.csv`
4. `wwr_trend_tests_<outcome>.csv` / `wwr_trend_coding_<outcome>.csv`
5. `contrasts_<outcome>.csv`（尤其是交互模型）
6. `evidence_*.png`

原因：
- 先确认模型能不能用
- 再读 fixed effects
- 然后看趋势编码和 simple effects
- 最后再用 evidence 图做解释和汇报

---

## 第五步：先看主 outcome
当前 main 分支推荐的主结果顺序：
1. `share_pct`
2. `share_logit`
3. `FC_share`
4. `fc_share_logit`
5. `FC_rate`
6. `tfd_y`
7. `ttff_y`
8. `fc_y`

这些最适合承担当前 paper-facing main 的 headline 结果。

---

## 第六步：哪些 outcome 更适合作为补充/机制支持
通常放后面再读，除非研究问题本身就盯着它们：
- `ffd_y`
- `mfd_y`
- `rff_y`
- `MPD`

更合适的定位是：
- exploratory
- mechanism-support
- supplement-facing

---

## 第七步：怎么正确使用 explanatory visuals
位置：
- `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm_visuals/`

用途：
- 解释型辅图
- 论文/汇报用说明图
- 帮助理解统计结果

注意：
- **不要**用它替代核心统计表

---

## 第八步：two-part models 什么时候看
位置：
- `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/two_part_models/`

适合在这些情况下纳入主线：
- scene-feature 问题本身是研究核心
- two-part 结构有明确理论意义
- 你需要在 allocation LMM 之外补一条解释线

如果 scene-feature 只是次要问题，通常更适合放补充材料。

---

## 第九步：overall 和 Experience 怎么分工
这是当前 `main` 很关键的一条规则：

- `overall` = 描述性背景
- `Experience` = 主显著性入口

也就是说：
- 不要只靠 `grouped_overall` 去建立主统计结论
- `overall` 更适合交代水平、分布、整体形态
- 主统计叙事优先从 `Experience` LMM 输出里建立

---

## 如果我要最短路线写汇报
按下面顺序读就够：
1. `研究输出_时间戳/README_研究输出说明.txt`
2. `研究输出_时间戳/01_QC后_AfterQC/01_描述性分析_Descriptive/grouped_experience/png/`
3. `.../allocation_lmm/groupvar_Experience/tables/model_family_index.csv`
4. `.../allocation_lmm/groupvar_Experience/tables/three_model_packet_summary.csv`
5. 进入主 outcome 对应的 `fixef_* / wwr_trend_* / contrasts_*`
6. 最后再看 explanatory PNG 和 two-part outputs

---

## 最简总结
如果你只记住 4 句话，记这 4 句：
1. 正式汇报优先从 QC 后轨道开始
2. 描述性结果先看 PNG，再对表
3. 显著性主线先看 Experience LMM，再看 explanatory visuals
4. exploratory outcome 默认当补充，除非研究问题明确以它们为主
