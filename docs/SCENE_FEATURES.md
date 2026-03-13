# scene_features.csv notes

## What downstream currently requires

### `scripts/merge_scene_features.py`
Minimum required columns:
- `participant_id`
- `scene_id`

Everything else is optional and will just be carried through into the merged analysis table.

### `scripts/model_scene_feature_mixed.py`
Requires from merged table:
- `participant_id`
- outcome columns (`TFD`, `TTFF`, `FC`) from AOI table
- at least one predictor column from:
  - `table_density`
  - `distance_to_table_center_m`
  - `table_center_offset_ratio`
  - `illum_lux`
  - `crowding_level`
  - `occlusion_ratio`
  - `aoi_coverage_ratio`
  - `non_table_aoi_coverage_ratio`
  - `WWR`

### `scripts/model_aoi_two_part.py`
Consumes the merged analysis table. The more scene/group predictors available, the richer the model table will be.

## Auto-generated columns

`scripts/generate_scene_features.py` can derive these from:
- scene folder name / structure
- background image size
- AOI JSON polygons
- `group_manifest.csv`

Auto columns:
- identifiers: `participant_id`, `scene_id`
- scene files: `scene_folder`, `background_image`, `aoi_json`
- image geometry: `image_width_px`, `image_height_px`, `image_area_px`
- AOI geometry: `aoi_class_count`, `aoi_polygon_count`, `aoi_total_area_px`, `aoi_coverage_ratio`
- table geometry: `table_polygon_count`, `table_area_px`, `table_density`, `table_area_ratio`, `table_center_x_px`, `table_center_y_px`, `table_center_offset_ratio`
- non-table AOI structure: `non_table_aoi_area_px`, `non_table_aoi_coverage_ratio`, `occlusion_ratio`, `crowding_level`, `non_table_class_count`, `has_table`
- parsed condition labels: `WWR`, `Complexity`, `condition_id`, `round`, `round_label`
- copied participant grouping fields when available: `SportFreq`, `Experience`

## Still user-provided / optional enrichments

Not reliably derivable from the current 4 raw inputs alone:
- `distance_to_table_center_m` in real-world meters
- `illum_lux`
- `noise_db`
- any manual semantic annotations not encoded in folder names or AOI polygons
