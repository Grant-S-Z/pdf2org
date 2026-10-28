"""Run MiniCPM-OCR's SDK pipeline against a local ollama server."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_SERVER = "http://127.0.0.1:11434"
DEFAULT_MODEL = "minicpm-ocr"
LAYOUT_FILES = ("inference.pdiparams", "inference.yml", "inference.json")
PAGE_DELIMITER = "\n\n-----\n\n"


class OcrError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def layout_model_dir() -> Path:
    return project_root() / "models" / "PP-DocLayoutV3_infer"


def resolve_layout_device(requested: str) -> str:
    value = (requested or "auto").strip().lower()
    if value != "auto":
        return value
    if not sys.platform.startswith("linux"):
        return "cpu"
    if shutil.which("nvidia-smi") is None:
        return "cpu"
    try:
        probe = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "cpu"
    if probe.returncode != 0 or "GPU" not in probe.stdout:
        return "cpu"

    from importlib.metadata import PackageNotFoundError, version

    try:
        version("paddlepaddle-gpu")
    except PackageNotFoundError:
        print(
            "检测到 NVIDIA GPU，但当前安装的是 CPU 版 Paddle，版面检测仍用 CPU。"
            "安装 paddlepaddle-gpu 后可启用 GPU。",
            file=sys.stderr,
        )
        return "cpu"
    return "cuda:0"


def _request_json(url: str, payload: dict | None = None, timeout: float = 5.0) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def check_ollama(server_url: str, model: str) -> None:
    base = server_url.rstrip("/")
    try:
        tags = _request_json(f"{base}/api/tags")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise OcrError(
            f"无法连接 ollama ({server_url}): {exc}\n"
            "先启动服务: brew services start ollama"
        ) from exc

    names = {item.get("name") for item in tags.get("models", []) if item.get("name")}
    if not names & {model, f"{model}:latest"}:
        available = ", ".join(sorted(names)) or "（无）"
        raise OcrError(
            f"ollama 中没有模型 {model!r}。现有模型: {available}\n"
            "按 README 的 Modelfile 方式导入，或修改 --model。"
        )

    try:
        info = _request_json(f"{base}/api/show", {"model": model})
    except (urllib.error.URLError, OSError, ValueError):
        return
    capabilities = info.get("capabilities") or []
    if capabilities and "vision" not in capabilities:
        raise OcrError(
            f"模型 {model!r} 不具备 vision 能力（capabilities={capabilities}），"
            "请确认导入时同时包含了 mmproj 视觉投影文件。"
        )


def ensure_layout_model(allow_download: bool = True) -> Path:
    directory = layout_model_dir()
    missing = [
        name
        for name in LAYOUT_FILES
        if not (directory / name).is_file() or (directory / name).stat().st_size == 0
    ]
    if not missing:
        return directory

    if not allow_download:
        raise OcrError(
            f"版面模型缺失: {directory}（缺 {', '.join(missing)}），"
            "去掉 --no-download 可自动下载。"
        )

    directory.mkdir(parents=True, exist_ok=True)
    from minicpmvocr.layouts.download import ensure_layout_model as sdk_ensure_layout_model

    print(f"下载 PP-DocLayoutV3 到 {directory} ...", file=sys.stderr)
    try:
        sdk_ensure_layout_model(directory)
    except Exception as exc:
        raise OcrError(f"下载 PP-DocLayoutV3 失败: {exc}") from exc
    return directory


def _parse_summary(stdout: str) -> dict:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        pass
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stdout[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def _page_count(source: Path) -> int | None:
    if source.suffix.lower() != ".pdf":
        return 1
    try:
        import pymupdf

        with pymupdf.open(source) as document:
            return int(document.page_count)
    except Exception:
        return None


def _completed_pages(run_dir: Path) -> int:
    output = run_dir / "output"
    if not output.is_dir():
        return 0
    return sum(1 for path in output.glob("*.md") if path.is_file())


def _layout_pages(run_dir: Path) -> int:
    tasks = run_dir / "runtime" / "page_tasks.jsonl"
    if not tasks.is_file():
        return 0
    try:
        with tasks.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


class _Progress:
    def __init__(self, total: int | None, stream=None) -> None:
        self.total = total
        self.stream = stream or sys.stderr
        self.enabled = self.stream.isatty()
        self.last_done = -1
        self.first_done_at: float | None = None

    @staticmethod
    def _clock(seconds: float) -> str:
        value = int(max(seconds, 0))
        return f"{value // 60:02d}:{value % 60:02d}"

    def update(
        self,
        done: int,
        elapsed: float,
        laid_out: int | None = None,
        final: bool = False,
    ) -> None:
        if self.total is not None:
            done = min(done, self.total)
            if laid_out is not None:
                laid_out = min(laid_out, self.total)
        if done > 0 and self.first_done_at is None:
            self.first_done_at = elapsed

        if not self.enabled:
            if done != self.last_done and (done > 0 or final):
                if self.total is None:
                    print(f"  已处理 {done} 页", file=self.stream)
                else:
                    print(f"  页 {done}/{self.total}", file=self.stream)
                self.last_done = done
            return

        width = 22
        if self.total is None:
            bar = "-" * width
            status = f"已处理 {done} 页"
            tail = ""
        else:
            filled = int((done / self.total) * width) if self.total else 0
            filled = max(0, min(filled, width))
            bar = "#" * filled + "-" * (width - filled)
            status = f"{done}/{self.total} 页"
            if done == 0:
                if laid_out:
                    tail = "OCR 处理中"
                else:
                    tail = "准备中" if elapsed < 3 else "处理中"
            elif done >= self.total:
                tail = ""
            elif done < 2 or self.first_done_at is None:
                tail = "估算中"
            else:
                span = max(elapsed - self.first_done_at, 0.001)
                per_page = span / (done - 1)
                tail = f"剩余 ~{self._clock(per_page * (self.total - done))}"

        line = f"  进度 [{bar}] {status}  {self._clock(elapsed)}"
        if tail:
            line += f"  {tail}"
        if laid_out is not None and self.total and 0 < laid_out < self.total:
            line += f"  (版面 {laid_out}/{self.total})"
        print("\r" + line + "\033[K", end="", file=self.stream, flush=True)
        self.last_done = done

    def finish(self, done: int, elapsed: float) -> None:
        if self.enabled:
            if done != self.last_done:
                self.update(done, elapsed, final=True)
            print(file=self.stream)
        else:
            self.update(done, elapsed, final=True)


def run_parse(
    source: Path,
    out_md: Path,
    *,
    server_url: str,
    model: str,
    workers: int,
    timeout: int,
    keep_work: bool = False,
    device: str = "auto",
) -> dict:
    root = project_root()
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    layout_device = resolve_layout_device(device)
    print(f"  版面设备: {layout_device}", file=sys.stderr)
    run_tag = f"{source.stem}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    command = [
        sys.executable,
        "-m",
        "minicpmvocr",
        "parse",
        str(source.resolve()),
        "-o",
        str(out_md.resolve()),
        "--backend",
        "ollama",
        "--server-url",
        server_url,
        "--model",
        model,
        "--device",
        layout_device,
        "--layout-model",
        str(layout_model_dir()),
        "--workers",
        str(workers),
        "--set",
        "engine.num_ctx=16384",
        "--set",
        "engine.max_model_len=16384",
        "--set",
        "engine.request_timeout_sec=600",
        "--set",
        "engine.keep_alive=30m",
        "--set",
        f"run.run_root={runs_root}",
        "--run-tag",
        run_tag,
        "--page-delimiter",
        PAGE_DELIMITER,
    ]

    run_dir = runs_root / run_tag
    total_pages = _page_count(source)
    progress = _Progress(total_pages)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _drain(stream, sink: list[str]) -> None:
        for line in stream:
            sink.append(line)

    started = time.time()
    proc = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    readers = [
        threading.Thread(target=_drain, args=(proc.stdout, stdout_lines), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, stderr_lines), daemon=True),
    ]
    for reader in readers:
        reader.start()

    timed_out = False
    try:
        while proc.poll() is None:
            elapsed = time.time() - started
            if timeout > 0 and elapsed > timeout:
                timed_out = True
                proc.kill()
                break
            progress.update(
                _completed_pages(run_dir),
                elapsed,
                laid_out=_layout_pages(run_dir),
            )
            time.sleep(0.3)
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for reader in readers:
            reader.join(timeout=5)

    elapsed = time.time() - started
    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)
    done = _completed_pages(run_dir)

    if timed_out:
        progress.finish(done, elapsed)
        raise OcrError(f"OCR 超时（超过 {timeout} 秒）: {source.name}")

    if proc.returncode != 0:
        progress.finish(done, elapsed)
        output = stderr.strip() or stdout.strip()
        tail = "\n".join(output.splitlines()[-15:])
        hint = f"\n中间产物: {run_dir}" if run_dir.exists() else ""
        raise OcrError(f"minicpmvocr 失败（退出码 {proc.returncode}）:\n{tail}{hint}")

    progress.finish(total_pages if total_pages is not None else done, elapsed)

    summary = _parse_summary(stdout)
    actual_run_dir = Path(summary.get("run_dir") or run_dir)
    if not keep_work:
        shutil.rmtree(actual_run_dir, ignore_errors=True)

    counts = (summary.get("infer_summary") or {}).get("counts") or {}
    return {
        "pages": summary.get("pages", 0),
        "blocks": counts.get("blocks", 0),
        "failed_blocks": counts.get("failed_blocks", 0),
        "elapsed": elapsed,
        "run_dir": actual_run_dir,
        "layout_device": layout_device,
    }
