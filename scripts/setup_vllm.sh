#!/usr/bin/env bash
# 创建 vLLM 环境并打上 MiniCPM-OCR 兼容补丁（与仓库 .venv 完全隔离）。
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${VLLM_VENV:-$HOME/vllm-env}"
uv_bin="${UV_BIN:-uv}"
python_bin="${PYTHON_BIN:-python3.12}"
patch_file="$root/patches/vllm-0.33-minicpm-ocr-transformers-5.21.patch"

"$uv_bin" venv "$venv" --python "$python_bin"
"$uv_bin" pip install --python "$venv/bin/python" vllm ninja
# vLLM 0.33 的依赖上限是 transformers<5.21，但 minicpm_ocr 架构从 5.21 才进入 transformers
"$uv_bin" pip install --python "$venv/bin/python" "transformers==5.21.0"

site="$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
patch -p1 -d "$site" < "$patch_file"

echo "vLLM 环境就绪: $venv"
echo "启动服务: scripts/serve_vllm.sh"
