#!/usr/bin/env bash
# sparse COLMAP points for frame 0 of the TRAIN cameras with the given (fixed) poses, Docker image
# colmap/colmap:latest (needs Docker), CPU SIFT, --memory limited.
#   colmap_init.sh <gt_dir>          e.g. work/data/gt/n3dv_coffee_martini_1352
# Option names differ between COLMAP 3.x (SiftExtraction.use_gpu) and >= 3.12/4 (FeatureExtraction.use_gpu): detected
# from `-h`.
set -euo pipefail
GT=${1:?gt dir}
PY=${PY:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
W=$GT/colmap
rm -rf "$W"; mkdir -p "$W/sparse_out" "$W/sparse_txt"
$PY "$HERE/colmap_model.py" write "$GT" "$W"
PARAMS=$($PY -c "import json;print(json.load(open('$W/model.json'))['camera_params'])")
D="docker run --rm --memory 8g --cpus 20 -u $(id -u):$(id -g) -e HOME=/w -v $W:/w colmap/colmap:latest colmap"
HELP_E=$($D feature_extractor -h 2>&1 || true)
HELP_M=$($D exhaustive_matcher -h 2>&1 || true)
ex_args=(); ma_args=()
if grep -q "FeatureExtraction.use_gpu" <<<"$HELP_E"; then ex_args+=(--FeatureExtraction.use_gpu 0); else ex_args+=(--SiftExtraction.use_gpu 0); fi
if grep -q "FeatureExtraction.num_threads" <<<"$HELP_E"; then ex_args+=(--FeatureExtraction.num_threads 20); elif grep -q "SiftExtraction.num_threads" <<<"$HELP_E"; then ex_args+=(--SiftExtraction.num_threads 20); fi
if grep -q "FeatureMatching.use_gpu" <<<"$HELP_M"; then ma_args+=(--FeatureMatching.use_gpu 0); else ma_args+=(--SiftMatching.use_gpu 0); fi
if grep -q "FeatureMatching.num_threads" <<<"$HELP_M"; then ma_args+=(--FeatureMatching.num_threads 20); elif grep -q "SiftMatching.num_threads" <<<"$HELP_M"; then ma_args+=(--SiftMatching.num_threads 20); fi
echo "colmap options: ${ex_args[*]} | ${ma_args[*]}"
t0=$(date +%s)
$D feature_extractor --database_path /w/db.db --image_path /w/images --ImageReader.camera_model PINHOLE \
   --ImageReader.single_camera 1 --ImageReader.camera_params "$PARAMS" --SiftExtraction.max_num_features 16384 \
   --SiftExtraction.peak_threshold 0.004 "${ex_args[@]}"       # lower peak threshold: ~2x keypoints (default gave 3.6K points)
$PY "$HERE/colmap_model.py" fixids "$GT" "$W"
$D exhaustive_matcher --database_path /w/db.db "${ma_args[@]}"
$D point_triangulator --database_path /w/db.db --image_path /w/images --input_path /w/sparse_in --output_path /w/sparse_out
$D model_converter --input_path /w/sparse_out --output_path /w/sparse_txt --output_type TXT
echo "colmap $(( $(date +%s) - t0 )) s"
$PY "$HERE/colmap_model.py" points "$GT" "$W"
rm -rf "$W/images"
