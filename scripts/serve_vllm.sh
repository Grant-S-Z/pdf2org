#!/usr/bin/env bash
# 用 vLLM 启动 MiniCPM-OCR（OpenAI 兼容），默认单核、GPU 1、端口 18499。
set -euo pipefail

model_dir="${1:-${MODEL_DIR:-$HOME/models/MiniCPM-OCR-1.0-hf}}"
port="${2:-${PORT:-18499}}"
venv="${VLLM_VENV:-$HOME/vllm-env}"
cpus="${CPUS:-0}"
gpu="${GPU_ID:-1}"

export PATH="$venv/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_VISIBLE_DEVICES="$gpu"

exec taskset -c "$cpus" "$venv/bin/vllm" serve "$model_dir" \
  --served-model-name minicpm-ocr \
  --port "$port" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.5 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --trust-request-mm-kwargs \
  --disable-log-requests
