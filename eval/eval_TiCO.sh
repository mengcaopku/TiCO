set -euo pipefail

MODEL_PATH=""
EVAL_SCRIPT="eval/scripts/eval_TiCO.py"
BENCH_ROOT="path/to/Eval_Dataset"
NPROC=8
MASTER_PORT=33099
MIN_PIXELS=3136

DATASETS=(
  "HR_Bench_4K"
  "HR_Bench_8K"
  "vstar_bench"
)

mkdir -p logs

for DATASET in "${DATASETS[@]}"; do
  EVAL_JSON="eval/dataset/${DATASET}.jsonl"
  OUT_DIR="eval/output/${DATASET}"
  OUT_IMAGE_DIR="eval/thinking_images/${DATASET}"
  OUT_FILE="${OUT_DIR}/TiCO.jsonl"
  DS_PATH="${BENCH_ROOT}/${DATASET}/"

  mkdir -p "${OUT_DIR}"
  mkdir -p "${OUT_IMAGE_DIR}"

  if [[ ! -f "${EVAL_JSON}" ]]; then
    echo "[WARN] ${EVAL_JSON} not exist, skip ${DATASET}."
    continue
  fi

  echo "========== Evaluating ${DATASET} =========="
  python -m torch.distributed.run \
      --nproc_per_node="${NPROC}" \
      --master_port="${MASTER_PORT}" \
      "${EVAL_SCRIPT}" \
      --model_path "${MODEL_PATH}" \
      --eval_dataset "${EVAL_JSON}" \
      --output_file "${OUT_FILE}" \
      --min_pixels "${MIN_PIXELS}" \
      --max_pixels 40000000 \
      --dataset_path "${DS_PATH}" \
      --dataset_name "${DATASET}" \
      --out_img_dir "${OUT_IMAGE_DIR}"
      2>&1 | tee "logs/${DATASET}.log"

done

