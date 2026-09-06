#!/usr/bin/env bash
set -Eeuo pipefail

# Run from anywhere; all generated paths are anchored at the repository root.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/sft_verifier/src:${REPO_ROOT}/overall_pipeline/src${PYTHONPATH:+:${PYTHONPATH}}"

CONDA_ENV="${CONDA_ENV:-rl}"
RUN_TAG="${RUN_TAG:-v1}"
COLLECT_GPUS="${COLLECT_GPUS:-2,3}"
TRAIN_GPUS="${TRAIN_GPUS:-2,3,4,5,6,7}"
PLANNER_MODEL="${PLANNER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PLANNER_REVISION="${PLANNER_REVISION:-989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
CODER_MODEL="${CODER_MODEL:-${REPO_ROOT}/sft/models/qwen2.5_0.5b_sql_coder}"
SAMPLES_PER_QUESTION="${SAMPLES_PER_QUESTION:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
EPOCHS="${EPOCHS:-2}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-16}"

IFS=',' read -r -a COLLECT_GPU_LIST <<< "${COLLECT_GPUS}"
IFS=',' read -r -a TRAIN_GPU_LIST <<< "${TRAIN_GPUS}"
if (( ${#COLLECT_GPU_LIST[@]} < 2 )); then
  echo "COLLECT_GPUS must contain at least two GPU IDs (planner,coder)." >&2
  exit 2
fi
if (( ${#TRAIN_GPU_LIST[@]} < 1 )); then
  echo "TRAIN_GPUS must contain at least one GPU ID." >&2
  exit 2
fi
if [[ ! -d "${CODER_MODEL}" ]]; then
  echo "Coder checkpoint does not exist: ${CODER_MODEL}" >&2
  exit 2
fi

MANIFEST="${REPO_ROOT}/sft_verifier/data/manifests/spider_train_splits.json"
CANDIDATE_DIR="${REPO_ROOT}/sft_verifier/data/candidates/${RUN_TAG}"
LABEL_DIR="${REPO_ROOT}/sft_verifier/data/private_labels/${RUN_TAG}"
SFT_DIR="${REPO_ROOT}/sft_verifier/data/sft/${RUN_TAG}"
MODEL_DIR="${REPO_ROOT}/sft_verifier/models/verifier_lora_${RUN_TAG}"
MERGED_DIR="${REPO_ROOT}/sft_verifier/models/verifier_merged_${RUN_TAG}"

mkdir -p "${CANDIDATE_DIR}" "${LABEL_DIR}" "${SFT_DIR}"

run_python() {
  conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

LIMIT_ARGS=()
if [[ -n "${MAX_EXAMPLES}" ]]; then
  LIMIT_ARGS=(--limit "${MAX_EXAMPLES}")
fi

echo "[1/7] Preparing DB-disjoint Spider split manifest"
run_python -m sft_verifier.prepare_splits --output "${MANIFEST}"

for SPLIT in train validation; do
  echo "[2/7] Collecting ${SPLIT} planner/coder candidates on GPUs ${COLLECT_GPUS}"
  CUDA_VISIBLE_DEVICES="${COLLECT_GPUS}" run_python -m sft_verifier.collect_candidates \
    --split-manifest "${MANIFEST}" \
    --split "${SPLIT}" \
    --planner-model "${PLANNER_MODEL}" \
    --planner-revision "${PLANNER_REVISION}" \
    --planner-device cuda:0 \
    --coder-model "${CODER_MODEL}" \
    --coder-device cuda:1 \
    --samples-per-question "${SAMPLES_PER_QUESTION}" \
    "${LIMIT_ARGS[@]}" \
    --resume \
    --output "${CANDIDATE_DIR}/${SPLIT}.jsonl"

  echo "[3/7] Creating ${SPLIT} single-edit hard negatives"
  run_python -m sft_verifier.mutate_gold \
    --candidates "${CANDIDATE_DIR}/${SPLIT}.jsonl" \
    --output "${CANDIDATE_DIR}/${SPLIT}_mutations.jsonl"

  echo "[4/7] Labeling ${SPLIT} candidates with the official Spider evaluator"
  run_python -m sft_verifier.label_candidates \
    --candidates \
      "${CANDIDATE_DIR}/${SPLIT}.jsonl" \
      "${CANDIDATE_DIR}/${SPLIT}_mutations.jsonl" \
    --output "${LABEL_DIR}/${SPLIT}.jsonl"
done

echo "[5/7] Building balanced, gold-reference-free SFT files"
run_python -m sft_verifier.build_sft_dataset \
  --labeled "${LABEL_DIR}/train.jsonl" "${LABEL_DIR}/validation.jsonl" \
  --output-dir "${SFT_DIR}"

echo "[6/7] Training verifier LoRA on GPUs ${TRAIN_GPUS}"
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" conda run --no-capture-output -n "${CONDA_ENV}" \
  torchrun --standalone --nproc_per_node="${#TRAIN_GPU_LIST[@]}" \
  -m sft_verifier.train_lora \
  --train-file "${SFT_DIR}/train.jsonl" \
  --validation-file "${SFT_DIR}/validation.jsonl" \
  --output-dir "${MODEL_DIR}" \
  --merge-output "${MERGED_DIR}" \
  --epochs "${EPOCHS}" \
  --effective-batch-size "${EFFECTIVE_BATCH_SIZE}"

echo "[7/7] Creating a multi-agent config for the merged verifier"
run_python -m sft_verifier.make_pipeline_config \
  --base-config "${REPO_ROOT}/overall_pipeline/configs/multi_turn_multi_agent_zero_shot.json" \
  --merged-verifier "${MERGED_DIR}" \
  --output "${REPO_ROOT}/sft_verifier/configs/multi_turn_sft_verifier_${RUN_TAG}.json"

echo "Completed verifier SFT run: ${RUN_TAG}"
echo "Merged verifier: ${MERGED_DIR}"
echo "Pipeline config: ${REPO_ROOT}/sft_verifier/configs/multi_turn_sft_verifier_${RUN_TAG}.json"
