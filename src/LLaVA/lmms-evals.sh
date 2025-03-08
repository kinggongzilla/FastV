#!/usr/bin/env bash

# Path to your fastv_kvcache.py file
KVCACHE_FILE="./llava/model/language_model/fastv_kvcache.py"

# Model Name
MODEL_NAME="llava-onevision-qwen2-0.5b-ov"

# If RANK is undefined
if [ -z "$RANK" ]; then
  # Set RANK to 0
  RANK=0
fi

# Python command (including arguments) that you want to run
# We'll place a placeholder for the --log_samples_suffix, which we’ll update for each (K, ratio) pair
RUN_CMD_BASE="python3 -m accelerate.commands.launch \
    --mixed_precision fp16 \
    --machine_rank=$RANK \
    --num_machines=1 \
    --num_processes=4 \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=\"../../../$MODEL_NAME/,conv_template=qwen_2\" \
    --tasks videomme \
    --batch_size 1 \
    --log_samples"

# List of (K, total_ratio, global_ratio) pairs you want to test
declare -a pairs=(
#  "1 0.9 0.9"
#  "1 0.75 0.75"
  "1 0.5 0.5"
  "1 0.25 0.25"
  "1 0.1 0.1"
#  "2 0.9 0.9"
#  "2 0.75 0.75"
  "2 0.5 0.5"
  "2 0.25 0.25"
  "2 0.1 0.1"
#  "3 0.9 0.9"
#  "3 0.75 0.75"
  "3 0.5 0.5"
  "3 0.25 0.25"
  "3 0.1 0.1"
  #  "3 0.5 0.1"
  #  "3 0.5 0.25"
  #  "3 0.5 0.75"
  #  "3 0.5 0.9"
  "100 1 1"
)

# Loop over each (K, total_ratio, global_ratio) pair
for pair in "${pairs[@]}"; do
  # Split pair into three variables
  set -- $pair
  KVAL="$1"
  TOTAL_RVAL="$2"
  GLOBAL_RVAL="$3"

  echo "========================================"
  echo "Running with K = $KVAL, total_ratio = $TOTAL_RVAL, global_ratio = $GLOBAL_RVAL"
  echo "========================================"

  # Use sed to replace lines in fastv_kvcache.py
  # Make sure these sed expressions match exactly how K=..., total_ratio=..., and global_ratio=... appear in your file
  sed -i "s/^K = .*/K = $KVAL/" "$KVCACHE_FILE"
  sed -i "s/^total_ratio = .*/total_ratio = $TOTAL_RVAL/" "$KVCACHE_FILE"
  sed -i "s/^global_ratio = .*/global_ratio = $GLOBAL_RVAL/" "$KVCACHE_FILE"

  # Construct the suffix for logs, for example k=3_r=0_5
  # We replace the dot (.) with underscore (_) for ratio
  TOTAL_RVAL_UNDERSCORE="${TOTAL_RVAL//./_}"
  GLOBAL_RVAL_UNDERSCORE="${GLOBAL_RVAL//./_}"
  LOG_SUFFIX="k${KVAL}_tr${TOTAL_RVAL_UNDERSCORE}_gr${GLOBAL_RVAL_UNDERSCORE}_${MODEL_NAME}"
  OUTPUT_PATH="./logs/${LOG_SUFFIX}"

  # Execute the command
  CMD="$RUN_CMD_BASE --log_samples_suffix $LOG_SUFFIX --output_path $OUTPUT_PATH"
  echo "Executing: $CMD"
  eval "$CMD"

  echo
done
