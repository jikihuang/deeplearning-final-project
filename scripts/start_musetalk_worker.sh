#!/usr/bin/env bash
set -e

cd /workspace/digital_human_demo
conda activate digitalhuman

export PATH=/usr/bin:$PATH
export MUSETALK_CUDA_VISIBLE_DEVICES=${MUSETALK_CUDA_VISIBLE_DEVICES:-1}
export MUSETALK_WORKER_PORT=${MUSETALK_WORKER_PORT:-8890}

python musetalk_worker_server.py
