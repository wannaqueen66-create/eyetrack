# RESULTS_MAP.md

眼动仓库 `main` 分支的一页速查：当前主线下，不同研究问题优先看哪些文件。

> 说明：
> - 当前 `main` 只强调两条结果轨道：`00_全样本_AllSample` 与 `01_QC后_AfterQC`
> - 正式汇报优先看 `01_QC后_AfterQC`
> - 如果只做结果浏览，建议优先看 `Experience` 显著性主线，而不是把 `overall` 描述性结果和主统计结论混成一条证据链

---

## 0. 总入口先看哪里？
1. `README.md`
2. `README_zh.md`
3. `docs/PROJECT_OVERVIEW.md`
4. `docs/RESULTS_READING_GUIDE.md`
5. `docs/RESULTS_READING_GUIDE.zh.md`
6. `docs/SIGNIFICANCE_MAINLINE.md`

---

## 1. 跑完后，整个结果文件夹先看哪里？
1. `研究输出_时间戳/README_研究输出说明.txt`
2. `研究输出_时间戳/01_QC后_AfterQC/README_研究输出说明.txt`
3. `研究输出_时间戳/01_QC后_AfterQC/01_描述性分析_Descriptive/`
4. `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/`

---

## 2. 全样本的描述性结果先看哪里？
1. `研究输出_时间戳/00_全样本_AllSample/01_描述性分析_Descriptive/grouped_overall/png/`
2. `研究输出_时间戳/00_全样本_AllSample/01_描述性分析_Descriptive/grouped_overall/tables/`
3. `研究输出_时间戳/00_全样本_AllSample/01_描述性分析_Descriptive/organized_outputs/`

---

## 3. Experience 分组的描述性差异先看哪里？
1. `研究输出_时间戳/01_QC后_AfterQC/01_描述性分析_Descriptive/grouped_experience/png/`
2. `研究输出_时间戳/01_QC后_AfterQC/01_描述性分析_Descriptive/grouped_experience/data/`
3. `研究输出_时间戳/01_QC后_AfterQC/01_描述性分析_Descriptive/grouped_experience/tables/`

重点：
- 先用 PNG 看模式
- 再用 `*_data.csv` / summary tables 对数值

---

## 4. 显著性主结果先看哪里？
1. `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/model_family_index.csv`
2. `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/three_model_packet_summary.csv`
3. `01_main_effects/`
4. `02_two_way_interactions/`
5. `03_three_way_interaction/`

---

## 5. 每个 outcome 在 LMM 文件夹里怎么读？
优先顺序：
1. `model_stability_summary.csv`
2. `model_fit_<outcome>.csv`
3. `fixef_<outcome>.csv`
4. `wwr_trend_tests_<outcome>.csv` / `wwr_trend_coding_<outcome>.csv`
5. `contrasts_<outcome>.csv`（主要用于交互模型）
6. `evidence_*.png`

一句话：
- 先确认模型能不能用
- 再看 fixed effects
- 然后再看 trend / contrasts / evidence 图

---

## 6. 哪些 outcome 是正文优先看的主指标？
优先读：
1. `share_pct`
2. `share_logit`
3. `FC_share`
4. `fc_share_logit`
5. `FC_rate`
6. `tfd_y`
7. `ttff_y`
8. `fc_y`

这些最适合承担当前 main 分支的 headline 结果。

---

## 7. 哪些 outcome 更适合补充/机制讨论？
优先当 supplement / exploratory：
- `ffd_y`
- `mfd_y`
- `rff_y`
- `MPD`

---

## 8. `allocation_lmm_visuals` 应该什么时候看？
看这里：
- `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm_visuals/png/`
- `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm_visuals/data/`

用途：
- 辅助解释
- 做论文/汇报的说明图
- 不替代核心统计表

---

## 9. two-part models 什么时候并入主线？
看这里：
- `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/two_part_models/`

使用原则：
- 当 scene-feature 问题本身是研究主问题时，可并入主文
- 否则默认放补充

---

## 10. `overall` 和 `Experience` 应该怎么分工？
- `overall`：主要服务于描述性背景、总体水平、整体分布
- `Experience`：当前显著性主线的优先入口

一句话规则：
- `overall` 负责背景
- `Experience LMM` 负责主统计结论

---

## 11. 正式汇报时最短阅读路径是什么？
1. `研究输出_时间戳/README_研究输出说明.txt`
2. `研究输出_时间戳/01_QC后_AfterQC/01_描述性分析_Descriptive/grouped_experience/png/`
3. `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/model_family_index.csv`
4. `研究输出_时间戳/01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/three_model_packet_summary.csv`
5. 进入主 outcome 对应的 `fixef_* / wwr_trend_* / contrasts_* / evidence_*`

---

## 12. 如果我要旧版更发散的结构怎么办？
当前这个 `RESULTS_MAP.md` 只服务于 clean `main` 主线。

如果你明确要：
- 历史 exploratory 输出
- 旧 research_bundle / analysis2 认知框架
- 更复杂的兼容路径

请切到：
- `raw` 分支

不要把 `raw` 的历史路径和当前 main 的主阅读面混在一起解释。
