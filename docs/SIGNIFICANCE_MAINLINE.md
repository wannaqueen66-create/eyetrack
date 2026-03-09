# Significance mainline on `main`

This note fixes the inferential reading order for the current `main` branch.
Use the same structure for both:

- `00_全样本_AllSample/02_显著性分析_Significance`
- `01_QC后_AfterQC/02_显著性分析_Significance`

## 1. What is the inferential mainline?

The current inferential mainline is:

- `allocation_lmm/`
  - especially `groupvar_Experience/`
  - and inside it, the three explicit LMM family folders:
    - `01_main_effects/`
    - `02_two_way_interactions/`
    - `03_three_way_interaction/`
  - plus the new WWR trend layer exported inside each family folder for primary outcomes:
    - `wwr_trend_tests_<outcome>.csv`
    - `wwr_trend_coding_<outcome>.csv`
    - `evidence_wwr_trend_terms_<outcome>.png`
    - `evidence_wwr_trend_shape_<outcome>.png`
- plus `allocation_lmm_visuals/` as explanatory companion figures
- plus `two_part_models/` only when scene-feature questions are central

`grouped_overall` belongs to the **descriptive** mainline, not the inferential mainline.
Use it for context and pattern description, not as the main statistical claim path.

## 2. Primary vs supplementary outcomes

### Primary / main-result outcomes
Recommended headline order:

1. `share_pct`
2. `share_logit`
3. `FC_share`
4. `fc_share_logit`
5. `FC_rate`
6. `tfd_y`
7. `ttff_y`
8. `fc_y`

Interpretive logic:
- `share_pct` / `share_logit`: TFD-based allocation
- `FC_share` / `fc_share_logit`: FC-based allocation
- `FC_rate`: time-standardized count intensity
- `tfd_y`: absolute attention amount
- `ttff_y`: latency to first entry
- `fc_y`: absolute fixation count

### Supplementary / exploratory outcomes
- `ffd_y`
- `mfd_y`
- `rff_y`
- `MPD`

These are useful for support, mechanism discussion, or supplement tables, but should not be the first headline unless they are the explicit study target.

## 3. Required reading order within each group folder

Recommended entry:

1. `model_family_index.csv`
2. `three_model_packet_summary.csv`
3. Then enter the three family folders in this order:
   - `01_main_effects/`
   - `02_two_way_interactions/`
   - `03_three_way_interaction/`
4. Within each family folder:
   - `model_stability_summary.csv`
   - `evidence_stability_overview_<GroupVar>.png`
   - `evidence_model_fit_overview_<GroupVar>.png`
   - For each primary outcome:
     - `model_fit_<outcome>.csv`
     - `fixef_<outcome>.csv`
     - `contrasts_<outcome>.csv` (interaction families first; main-effects family may omit or de-emphasize this)
     - `evidence_fixef_key_terms_<outcome>.png`
     - `evidence_contrasts_<outcome>.png` when contrasts are exported
   - `ranef_<outcome>.csv` when variance decomposition is needed
   - `forest_fixef_<outcome>.png` for compact audit / appendix support

## 4. How the main evidence pieces map to each other

For one outcome, treat the files as one packet:

- `model_fit_<outcome>.csv`
  - answers: did the model fit, what is the sample base, what is the approximate R², is it converged/stable?
- `fixef_<outcome>.csv`
  - answers: which fixed effects are estimated, with coefficient / SE / z / p / CI
- `contrasts_<outcome>.csv`
  - answers: what are the simple effects around WWR × Complexity × Group, in reviewer-friendly form
- `wwr_trend_tests_<outcome>.csv`
  - answers: is the ordered WWR pattern mainly linear up/down, or quadratic with a mid-point peak/trough?
- `wwr_trend_coding_<outcome>.csv`
  - answers: how each observed WWR level was encoded for the linear and quadratic tests
- `evidence_fixef_key_terms_<outcome>.png`
  - visual summary of key fixed effects
- `evidence_wwr_trend_terms_<outcome>.png`
  - compact coefficient CI plot for linear/quadratic WWR terms
- `evidence_wwr_trend_shape_<outcome>.png`
  - compact shape sketch that makes “linear increase/decrease” vs “45 highest/lowest” visually explicit
- `evidence_contrasts_<outcome>.png`
  - visual summary of contrasts
- `*_data.csv`
  - exact exported data behind the PNGs

So the practical workflow is:

- use `three_model_packet_summary.csv` to decide which model family and outcome are eligible for the main text
- within a family, use `model_stability_summary.csv` to check stability before quoting effects
- use `model_fit_<outcome>.csv` + `fixef_<outcome>.csv` for the core statistical claim in that family
- use `contrasts_<outcome>.csv` mainly for the two-way / three-way interaction families when you need simple-effect wording and reviewer response support
- use evidence PNGs for compact communication in manuscript drafting / supplement / rebuttal

## 5. Experience vs overall

On `main`, keep this separation clear:

- `overall` = descriptive path
- `Experience` LMM = inferential main path
- `SportFreq` LMM = secondary inferential path

If a result is introduced from `grouped_overall`, phrase it as descriptive background.
If a result is introduced as a statistical claim, route it through the LMM outputs.

## 6. Stability rule for manuscript use

- `stable` → default main-result candidate
- `caution` → usable with explicit caution language
- `unstable` → supplement / diagnostic unless the model is reworked

For paper writing and reviewer response, do not quote a strong inferential claim before checking stability first.
