# Calibration target presets

Tracked target geometry belongs under `assets/`; capture results and generated
diagnostics belong under `outputs/`.

Available physical presets:

- `charuco/charuco_7x5_40mm_marker30mm_DICT_5X5_50.yaml`: A4 landscape
  ChArUco board, 7x5 squares, 40 mm squares, 30 mm markers, `DICT_5X5_50`.
- `../apriltag_grid/compact_apriltag_grid_4x4_tag48mm.yaml`: 4x4
  `tag36h11` grid with 48 mm tags.

Each separately printed board must receive a distinct `physical_id`, even when
it uses the same geometry preset.
