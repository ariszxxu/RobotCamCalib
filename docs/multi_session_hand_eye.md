# Generic multi-session hand-eye calibration

`solve_multi_session_hand_eye.py` fits one shared mount-to-camera transform
from two or more capture sessions. Each session receives an independent fixed
target pose, so a small target movement between sessions does not become a
camera-extrinsic error.

Before solving, the tool rereads every configured image, perspective-normalizes
the planar target ROI, calculates a Tenengrad plus Laplacian focus score, rejects
configured reprojection failures, and robustly rejects only the low-sharpness
tail within each session. Every image receives a score, status, and reason in
the output YAML.

The rig adapter is a YAML file. It defines accepted input schemas, dotted paths
to the sample list and transform fields, hardware/target consistency fields,
planar target size, image-quality thresholds, solver scales, cross-validation,
and the output transform key. See
`configs/multi_session_hand_eye_xarm7_g305.yaml` for a complete example.

Example using three raw capture manifests:

```bash
python solve_multi_session_hand_eye.py \
  outputs/extrinsics/xarm7_g305_eye_in_hand/samples/SESSION_A/capture_manifest.yaml \
  outputs/extrinsics/xarm7_g305_eye_in_hand/samples/SESSION_B/capture_manifest.yaml \
  outputs/extrinsics/xarm7_g305_eye_in_hand/samples/SESSION_C/capture_manifest.yaml \
  --config configs/multi_session_hand_eye_xarm7_g305.yaml \
  --output outputs/extrinsics/xarm7_g305_eye_in_hand/multi_session_candidate.yaml
```

The output remains `candidate_requires_physical_validation`; promotion to
`final` is an explicit operator decision. Keep separate adapter files for rigs
whose schema, frames, camera model, target geometry, or thresholds differ.
