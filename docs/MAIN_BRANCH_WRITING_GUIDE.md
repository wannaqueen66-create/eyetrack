# Main-branch Writing Guide (eyetrack)

这份提纲服务于当前眼动仓库 `main` 分支，不走旧 exploratory / raw 阅读面。

## 一、推荐的正文结果顺序

1. 先交代双轨结构：全样本 + QC后
2. 正式正文优先以 `01_QC后_AfterQC` 为主
3. 先写 Experience 描述性模式
4. 再写 allocation LMM 的 Experience 主显著性链
5. 再补 explanatory visuals
6. 若 scene features 是主问题，再把 two-part models 并入主文；否则放补充

## 二、推荐先引用的关键文件

### 结果总说明
- `研究输出_时间戳/README_研究输出说明.txt`
- `研究输出_时间戳/01_QC后_AfterQC/README_研究输出说明.txt`

### 描述性主入口
- `01_QC后_AfterQC/01_描述性分析_Descriptive/grouped_experience/png/`
- `01_QC后_AfterQC/01_描述性分析_Descriptive/grouped_experience/data/`

### 显著性主入口
- `01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/model_family_index.csv`
- `01_QC后_AfterQC/02_显著性分析_Significance/allocation_lmm/groupvar_Experience/tables/three_model_packet_summary.csv`

## 三、每个 outcome 的推荐写法顺序

1. `model_fit_<outcome>.csv`：先确认模型可用
2. `fixef_<outcome>.csv`：再写主效应/交互
3. `wwr_trend_tests_<outcome>.csv` / `wwr_trend_coding_<outcome>.csv`：补趋势方向
4. `contrasts_<outcome>.csv`：写 simple effects 或组间差异
5. `evidence_*.png`：辅助解释、放图注或汇报材料

## 四、正文优先 outcome

按当前 main 主线，优先：
- `share_pct`
- `share_logit`
- `FC_share`
- `fc_share_logit`
- `FC_rate`
- `tfd_y`
- `ttff_y`
- `fc_y`

## 五、默认放补充的 outcome

- `ffd_y`
- `mfd_y`
- `rff_y`
- `MPD`

## 六、图和表的分工

- 图：讲模式、方向、相对强弱
- 表：给具体估计值、检验值、p 值、趋势编码、contrast 细节
- 不要让 explanatory visuals 取代核心统计表

## 七、推荐的正文段落骨架

1. 样本与 QC 口径
2. Experience 描述性结果
3. Experience allocation LMM 主结果
4. 关键 trend / contrast 结果
5. exploratory 指标或 two-part 扩展结果（如需要）
6. 全样本结果作为稳健性/背景对照
