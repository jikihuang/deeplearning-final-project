#!/usr/bin/env bash
set -e

echo "== Check files =="
test -f /workspace/digital_human_demo/app.py && echo "app.py OK"
test -f /workspace/digital_human_demo/musetalk_worker_server.py && echo "worker OK"
test -f /workspace/digital_human_demo/.env && echo ".env OK"

echo "== Check Boson env =="
grep -n "BOSON_API_KEY\|BOSON_VOICE" /workspace/digital_human_demo/.env || true

echo "== Check Qwen =="
curl -s http://127.0.0.1:8001/v1/models || true
echo

echo "== Check MuseTalk Worker =="
curl -s http://127.0.0.1:8890/health || true
echo

echo "== Check ports =="
ss -ltnp | grep -E "8001|8890|8888" || true
