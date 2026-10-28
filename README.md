# pdf2org

用本地 ollama 上的 MiniCPM-OCR 把 PDF / 图片转成 Org 或 Markdown。版面检测、OCR、格式转换全部在本机完成，不调用云端服务。

## 特性

- 输入 PDF 或图片，输出 `.org` / `.md` / 两者
- PP-DocLayoutV3 做版面分析，按阅读顺序切块
- 逐块调用本地 ollama 模型，支持中英文、表格、公式、印章、图表
- Markdown → Org 保留原生表格与 LaTeX 公式，不产生 `#+begin_export html`
- 批量处理目录，终端两段式进度条（版面 + OCR）
- 失败时保留中间产物，便于排查

## 环境要求

- macOS（Apple Silicon 测试通过）或 Linux
- Python 3.12（由 uv 管理）
- [uv](https://docs.astral.sh/uv/)
- [ollama](https://ollama.com/) 及 `minicpm-ocr` 模型
- [pandoc](https://pandoc.org/) 3.x
- 建议 16 GB 内存

以下安装命令以 macOS 为例。

## 安装

### 1. 外部工具

```bash
brew install uv ollama pandoc
brew services start ollama
```

### 2. 准备 OCR 模型

ollama 官方模型库还没有 `minicpm-ocr`，需要用官方 GGUF + Modelfile 导入。从 ModelScope（国内较快）或 HuggingFace 下载两个文件：

```bash
mkdir -p ~/models/minicpm-ocr && cd ~/models/minicpm-ocr

curl -L -O https://modelscope.cn/models/OpenBMB/MiniCPM-OCR-1.0-gguf/resolve/master/MiniCPM-OCR-1_0-F16.gguf
curl -L -O https://modelscope.cn/models/OpenBMB/MiniCPM-OCR-1.0-gguf/resolve/master/mmproj-model-f16-4x.gguf
```

创建 `Modelfile`：

```dockerfile
FROM ./MiniCPM-OCR-1_0-F16.gguf
FROM ./mmproj-model-f16-4x.gguf

PARAMETER temperature 0
PARAMETER num_ctx 16384
```

导入并确认具备 `vision` 能力：

```bash
ollama create minicpm-ocr -f Modelfile
ollama show minicpm-ocr
```

### 3. 项目依赖

```bash
git clone <仓库地址> pdf2org
cd pdf2org
uv sync
```

首次运行时自动下载 PP-DocLayoutV3 版面模型（约 126 MB，带 SHA256 校验）到 `models/PP-DocLayoutV3_infer`。

## 使用

```bash
./pdf2org report.pdf                     # 默认输出 out/report.org
./pdf2org report.pdf -f md               # 只输出 Markdown
./pdf2org report.pdf -f both -o out      # 两者都要
./pdf2org ./docs -f org -o out           # 递归处理目录中的 PDF
./pdf2org report.pdf --force             # 覆盖已有输出
```

没有启动脚本时等价于：

```bash
uv run pdf2org report.pdf
```

### 常用选项

| 选项 | 默认 | 说明 |
|---|---|---|
| `INPUT...` | 必填 | PDF/图片文件或目录 |
| `-f, --format` | `org` | `org` / `md` / `both` |
| `-o, --out-dir` | `out` | 输出目录 |
| `--force` | 关 | 覆盖已有输出 |
| `--server-url` | `http://127.0.0.1:11434` | ollama 地址 |
| `--model` | `minicpm-ocr` | ollama 模型名 |
| `--workers` | `2` | OCR 并发块数 |
| `--timeout` | `0` | 单文件总超时秒数，0 为不限制 |
| `--device` | `auto` | 版面检测设备：`auto` / `cpu` / `cuda:0` |
| `--keep-work` | 关 | 保留 `runs/` 中间产物 |
| `--no-download` | 关 | 版面模型缺失时直接报错 |

### GPU 加速

OCR 推理由 ollama 自动使用 GPU（Metal / CUDA / ROCm），不需要本项目配置。运行后用 `ollama ps` 查看 `PROCESSOR` 列，`100% GPU` 表示全部在 GPU。

版面检测默认 `--device auto`：

- macOS：Paddle 没有 Metal 后端，始终用 CPU；
- Linux + NVIDIA：检测到 `nvidia-smi` 且安装了 `paddlepaddle-gpu` 时自动用 `cuda:0`，否则回退 CPU 并提示。

Linux 手动启用 GPU 版面（以 CUDA 12.9 为例）：

```bash
uv pip uninstall paddlepaddle
uv pip install --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ "paddlepaddle-gpu==3.3.0"

# uv sync 会把 CPU 版装回来；用 --no-sync 跳过同步
PDF2ORG_NO_SYNC=1 ./pdf2org --device auto doc.pdf
```

验证：

```bash
uv run --no-sync python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"
ollama ps
```

注意：`uv sync` 会按 `pyproject.toml` 重新安装 CPU 版 Paddle，覆盖手动安装的 GPU 版。需要长期使用 GPU 版面时，建议把 `paddlepaddle` 替换为 `paddlepaddle-gpu` 并配置 Paddle 的 CUDA 索引，或每次安装后使用 `--no-sync` / `PDF2ORG_NO_SYNC=1`。

### 输出说明

- `.md`：SDK 原始 Markdown，表格可能是 HTML。
- `.org`：经 pandoc 两段式转换，HTML 表格变成原生 `| ... |` 表格，公式保留为 `\(...\)` / `\[...\]`。
- 多页之间以 `-----` 水平线分隔。
- 标题层级由模型判断，页面标题常为 `**`（二级），不是固定的一级。

## 工作原理

```
PDF ──PyMuPDF──► 页面图 ──PP-DocLayoutV3──► 版面块
                                      │
                                      ▼
                              ollama 逐块识别
                                      │
                                      ▼
                              Markdown ──pandoc──► Org
```

- 版面检测默认自动选择设备：Linux + NVIDIA + GPU 版 Paddle 时用 CUDA，其余用 CPU；通过 PaddleX 加载 PP-DocLayoutV3。
- 识别请求走 ollama `/api/chat`，每个块单独请求。
- 进度条每 0.3 秒轮询 `runs/<tag>/output/*.md`（OCR 完成页）和 `runtime/page_tasks.jsonl`（版面已处理页）。
- 成功后删除 `runs/<tag>`；失败或 `--keep-work` 时保留。

## 项目结构

```
pdf2org/
├── pdf2org                 # 启动脚本
├── pyproject.toml
├── uv.lock
├── src/pdf2org/
│   ├── cli.py              # 参数、批量、预检、汇总
│   ├── ocr.py              # SDK 调用、进度条、模型检查
│   └── convert.py          # Markdown → Org
├── models/                 # 版面模型（自动下载，gitignore）
├── runs/                   # 中间产物（gitignore）
└── out/                    # 默认输出（gitignore）
```

## 限制

- SDK 固定以 144 dpi 渲染 PDF；小字号扫描件可能识别不佳。
- 不做跨页表格合并，不导出图片/图表裁剪。
- 进度只到“页”级别，块级进度 SDK 未暴露。
- 单个请求仍有 600 秒超时；`--timeout` 控制整份文件。
- 默认安装 CPU 版 Paddle；版面 GPU 需按“GPU 加速”一节手动安装 GPU 版。
- 大文档耗时较长，建议保持 ollama 服务运行，避免同时跑多个 OCR 任务。

## 排障

| 现象 | 处理 |
|---|---|
| `无法连接 ollama` | `brew services start ollama` |
| `ollama 中没有模型 'minicpm-ocr'` | 按“准备 OCR 模型”重新 `ollama create` |
| `找不到 pandoc` | `brew install pandoc` |
| 版面模型下载失败 | 重试；或手动放到 `models/PP-DocLayoutV3_infer` 后加 `--no-download` |
| 识别很慢 | `ollama ps` 确认模型已加载；检查内存压力/swap，关闭占内存的应用 |
| 输出不完整 | 查看失败时打印的 `runs/<tag>`，或用 `--keep-work` 保留中间产物 |

## 测试环境

macOS 27 / Apple Silicon / 16 GB，ollama 0.40.11，Python 3.12，pandoc 3.12。2 页测试文档热启动约 7 秒，具体取决于硬件和文档复杂度。
