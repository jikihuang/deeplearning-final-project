#!/usr/bin/env bash
set -e

cd /workspace/digital_human_demo
conda activate digitalhuman

export PATH=/usr/bin:$PATH
export MUSETALK_WORKER_URL=${MUSETALK_WORKER_URL:-http://127.0.0.1:8890}
export GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT:-8888}

python app.py
