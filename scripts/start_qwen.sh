#!/usr/bin/env bash
set -e

conda activate qwenllm

export HF_HOME=/workspace/hf_cache
export CUDA_VISIBLE_DEVICES=0

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.55 \
  --max-model-len 4096 \
  --served-model-name qwen2.5-7b-instruct \
  --api-key local-qwen \
  --trust-remote-code
