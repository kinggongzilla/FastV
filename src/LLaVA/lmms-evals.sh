#!/usr/bin/env bash

# Path to your fastv_kvcache.py file
KVCACHE_FILE="./llava/model/language_model/fastv_kvcache.py"

# Python command (including arguments) that you want to run
# We'll place a placeholder for the --log_samples_suffix, which we’ll update for each (K, ratio) pair
RUN_CMD_BASE="python3 -m accelerate.commands.launch \
    --mixed_precision fp16 \
    --num_processes=1 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=\"../../../llava-onevision-qwen2-0.5b-ov/,conv_template=qwen_2\" \
    --tasks flickr30k,nocaps,ok_vqa,mmmu \
    --batch_size 1 \
    --log_samples"

# RUN_CMD_BASE="python3 \
#     -m lmms_eval \
#     --model llava_onevision \
#     --model_args pretrained=\"../../../llava-onevision-qwen2-0.5b-ov/,conv_template=qwen_2,device_map=auto\" \
#     --tasks flickr30k,nocaps,ok_vqa,mmmu \
#     --batch_size 1 \
#     --log_samples"

# List of (K, ratio) pairs you want to test
declare -a pairs=(
  "1 0.9"
  "1 0.75"
  "1 0.5"
  "1 0.25"
  "1 0.1"
  "2 0.9"
  "2 0.75"
  "2 0.5"
  "2 0.25"
  "2 0.1"
  "3 0.9" 
  "3 0.75"
  "3 0.5"
  "3 0.25"
  "3 0.1"
  "100 1"
)

# Loop over each (K, ratio) pair
for pair in "${pairs[@]}"; do
  # Split pair into two variables
  set -- $pair
  KVAL="$1"
  RVAL="$2"

  echo "========================================"
  echo "Running with K = $KVAL, ratio = $RVAL"
  echo "========================================"

  # Use sed to replace lines in fastv_kvcache.py
  # Make sure these sed expressions match exactly how K=... and ratio=... appear in your file
  sed -i "s/^K = .*/K = $KVAL/" "$KVCACHE_FILE"
  sed -i "s/^ratio_global = .*/ratio_global = $RVAL/" "$KVCACHE_FILE"
  sed -i "s/^ratio_local = .*/ratio_local = $RVAL/" "$KVCACHE_FILE"

  # Construct the suffix for logs, for example k=3_r=0_5
  # We replace the dot (.) with underscore (_) for ratio
  RVAL_UNDERSCORE="${RVAL//./_}"
  LOG_SUFFIX="k${KVAL}_r${RVAL_UNDERSCORE}"
  OUTPUT_PATH="./logs/${LOG_SUFFIX}"

  # Execute the command
  CMD="$RUN_CMD_BASE --log_samples_suffix $LOG_SUFFIX --output_path $OUTPUT_PATH"
  echo "Executing: $CMD"
  eval "$CMD"

  echo
done
