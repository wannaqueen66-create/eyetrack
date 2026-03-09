# Metrics Specification / 指标命名与口径规范

This file defines the canonical AOI metric names for `eyetrack`.
本文件定义 `eyetrack` 主仓的标准 AOI 指标命名与计算口径。

## Canonical abbreviations / 标准英文缩写

- **FC** = Fixation Count
- **FFD** = First Fixation Duration
- **MFD** = Mean Fixation Duration
- **MPD** = Mean Pupil Diameter
- **RFF** = Re-fixation Frequency
- **TFD** = Total Fixation Duration
- **TTFF** = Time to First Fixation
- **FC_share / FC_prop** = AOI FC divided by total FC within the same trial
- **FC_rate** = AOI FC per second of trial duration
- **share / share_pct** = AOI TFD divided by total TFD within the same trial (raw proportion / percent)

## Canonical long names / 标准英文全称

- `Fixation Count`
- `First Fixation Duration`
- `Mean Fixation Duration`
- `Mean Pupil Diameter`
- `Re-fixation Frequency`
- `Total Fixation Duration`
- `Time to First Fixation` (`TTFF`)

## Legacy aliases still seen in helper scripts / 仍可能在辅助脚本中见到的旧别名

- `fixation_count` → `FC`
- `dwell_time_ms` → `TFD`
- `RF` → `RFF`

Canonical first-fixation timing output is now `TTFF` only.
首次注视时间输出现统一为 `TTFF`，不再输出旧别名列。

## Definitions / 定义

### FC
Number of unique fixation indices inside an AOI.
AOI 内唯一 fixation index 的数量。

### FFD
Duration of the first fixation that entered the AOI.
首次进入 AOI 的 fixation 的持续时间。

### MFD
Mean fixation duration across unique fixations inside the AOI.
AOI 内所有唯一 fixation 的平均注视时长。

### MPD
Mean pupil diameter in the AOI subset.
AOI 子集中的平均瞳孔直径。

### RFF
Number of AOI re-entry episodes after the first entry, based on fixation sequence.
基于 fixation 序列，首次进入之后重新进入 AOI 的次数。

### TFD
Total fixation duration in the AOI.
AOI 总注视时长。

### TTFF
Time from trial start to the first fixation entering the AOI.
从 trial 起点到首次进入 AOI 的时间。

### FC_share / FC_prop
AOI fixation count divided by the total fixation count within the same trial.
该 AOI 的 fixation count 除以同一 trial 的总 fixation count。

Use this when the question is about **allocation / proportion of attention episodes**, especially under unbalanced group sizes.
当研究问题是**注意分配/注视次数占比**，尤其面对组间样本量不平衡时，优先看该指标。

### FC_rate
AOI fixation count divided by trial duration in seconds.
该 AOI 的 fixation count 除以 trial 时长（秒）。

This is a rate-style standardization and is preferable to manually dividing raw FC by group size.
这是按时长率化的标准化指标，优于手工按组人数去除原始 FC。

### share / share_pct
AOI TFD divided by total TFD within the same trial (raw proportion / percent).
该 AOI 的 TFD 除以同一 trial 总 TFD（原始比例 / 百分比）。

Relationship among allocation metrics / 分配类指标之间的关系：
- `share_pct` / `share` describe **time allocation** (how much fixation duration the AOI captured within a trial).
- `FC_share` / `FC_prop` describe **count allocation** (how many fixation episodes the AOI captured within a trial).
- `FC_rate` describes **rate per unit time** rather than within-trial share.
- `TFD` and `FC` remain absolute magnitude metrics, useful as support, but they should not be manually normalized by group size.
- `TTFF` is an entry-latency metric; it answers a different question from share/rate metrics.

## Recommended operational settings / 推荐计算口径

- `point_source=fixation`
- `dwell_mode=fixation`

These settings should be treated as the default analysis settings for formal reports.
正式分析与论文报告默认采用以上口径。
