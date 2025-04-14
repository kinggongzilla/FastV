#!/usr/bin/env bash
# conda activate jm
# cd /system/user/publicwork/lin/lmm_repos/FastV_david/FastV/src/LLaVA/

# Path to your fastv_kvcache.py file
KVCACHE_FILE="./llava/model/language_model/fastv_kvcache.py"

# Model Name
MODEL_NAME="llava-onevision-qwen2-7b-ov"
MODEL_PATH="/system/user/publicdata/llm/Llava_weights/llava-onevision-qwen2-7b-ov"
OUTPUT_MAIN_DIR="/system/user/publicwork/lin/FastV_results"
#DATASET="ok_vqa"
#DATASET="mmmu_val"
#DATASET="mvbench"
#DATASET="coco2017_cap_val"
#DATASET="mlvu_test"
DATASET="textcaps_val"

#SAMPLING_MODE="Uniform"
SAMPLING_MODE="TokenWiseKVCompress"
gpu_id=1

NUM_PROCESSES=4

# If RANK is undefined
if [ -z "$RANK" ]; then
  # Set RANK to 0
  RANK=0
fi


# List of (K, total_ratio, global_ratio) pairs you want to test
declare -a pairs=(
##  "1 0.9 0.9"
##  "1 0.75 0.75"
#  "1 0.5 0.5"
#  "1 0.25 0.25"
#  "1 0.1 0.1"
#  "100 1 1"
#  "2 0.9 0.9"
#  "2 0.75 0.75"
  "0 0.5 0.5"
  "1 0.5 0.5"
   "2 0.5 0.5"
  "2 0.25 0.25"
  "2 0.1 0.1"

  "0 0.25 0.25"
  "0 0.1 0.1"

  "1 0.25 0.25"
  "1 0.1 0.1"


#  "0 0.25 0.25"
#  "0 0.1 0.1"
#  "0 0.05 0.05"
#  "0 0.01 0.01"
##  "3 0.9 0.9"
##  "3 0.75 0.75"
#  "3 0.5 0.5"
#  "3 0.25 0.25"
#  "3 0.1 0.1"
#  #  "3 0.5 0.1"
#  #  "3 0.5 0.25"
#  #  "3 0.5 0.75"
#  #  "3 0.5 0.9"

)

# Loop over each (K, total_ratio, global_ratio) pair
for pair in "${pairs[@]}"; do
  # Split pair into three variables
  set -- $pair
  KVAL="$1"
  TOTAL_RVAL="$2"
  GLOBAL_RVAL="$3"


  # Python command (including arguments) that you want to run
  # We'll place a placeholder for the --log_samples_suffix, which we’ll update for each (K, ratio) pair
  RUN_CMD_BASE="CUDA_VISIBLE_DEVICES=$gpu_id python3 -m accelerate.commands.launch \
      --mixed_precision fp16 \
      --num_machines=1 \
      --num_processes=$NUM_PROCESSES \
      -m lmms_eval \
      --model llava_onevision \
      --model_args pretrained=\"$MODEL_PATH,conv_template=qwen_2,sampling_mode=$SAMPLING_MODE,sampling_start_layer=$KVAL,keep_ratio=$TOTAL_RVAL\" \
      --tasks $DATASET \
      --batch_size 1 \
      --log_samples"


  echo "========================================"
  echo "Running with K = $KVAL, total_ratio = $TOTAL_RVAL, global_ratio = $GLOBAL_RVAL"
  echo "========================================"

  # Use sed to replace lines in fastv_kvcache.py
  # Make sure these sed expressions match exactly how K=..., total_ratio=..., and global_ratio=... appear in your file
#  sed -i "s/^K = .*/K = $KVAL/" "$KVCACHE_FILE"
#  sed -i "s/^total_ratio = .*/total_ratio = $TOTAL_RVAL/" "$KVCACHE_FILE"
#  sed -i "s/^global_ratio = .*/global_ratio = $GLOBAL_RVAL/" "$KVCACHE_FILE"
#  sed -i "s/^SAMPLING_MODE = .*/SAMPLING_MODE = \"$SAMPLING_MODE\"/" "$KVCACHE_FILE"

  # Construct the suffix for logs, for example k=3_r=0_5
  # We replace the dot (.) with underscore (_) for ratio
  TOTAL_RVAL_UNDERSCORE="${TOTAL_RVAL//./_}"
  GLOBAL_RVAL_UNDERSCORE="${GLOBAL_RVAL//./_}"
  LOG_SUFFIX="k${KVAL}_tr${TOTAL_RVAL_UNDERSCORE}_gr${GLOBAL_RVAL_UNDERSCORE}_${MODEL_NAME}_${SAMPLING_MODE}"
  OUTPUT_PATH="${OUTPUT_MAIN_DIR}/${LOG_SUFFIX}"

  # Execute the command
  CMD="$RUN_CMD_BASE --log_samples_suffix $LOG_SUFFIX --output_path $OUTPUT_PATH"
  echo "Executing: $CMD"
  eval "$CMD"

  echo
done
