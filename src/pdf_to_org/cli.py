"""pdf2org command line entry point."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from .convert import ConversionError, md_to_org
from .ocr import (
    DEFAULT_MODEL,
    DEFAULT_SERVER,
    OcrError,
    check_ollama,
    ensure_layout_model,
    run_parse,
)

SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return number


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须是 0 或正整数")
    return number


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2org",
        description="用本地 MiniCPM-OCR + ollama 把 PDF 转成 Org 或 Markdown。",
    )
    parser.add_argument("inputs", nargs="+", metavar="INPUT", help="PDF/图片文件，或包含 PDF 的目录")
    parser.add_argument(
        "-f",
        "--format",
        choices=("org", "md", "both"),
        default="org",
        help="输出格式，默认 org",
    )
    parser.add_argument("-o", "--out-dir", default="out", help="输出目录，默认 ./out")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的输出")
    parser.add_argument("--server-url", default=DEFAULT_SERVER, help=f"ollama 地址，默认 {DEFAULT_SERVER}")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"ollama 模型名，默认 {DEFAULT_MODEL}")
    parser.add_argument("--workers", type=_positive_int, default=2, help="OCR 并发块数，默认 2")
    parser.add_argument(
        "--timeout",
        type=_non_negative_int,
        default=0,
        help="单文件超时秒数，0 表示不限制（默认 0）",
    )
    parser.add_argument("--keep-work", action="store_true", help="保留 runs/ 下的中间产物")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="版面模型缺失时直接报错，不自动下载",
    )
    return parser


def _discover(raw_inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in raw_inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            files.extend(
                sorted(
                    item
                    for item in path.rglob("*")
                    if item.is_file() and item.suffix.lower() == ".pdf"
                )
            )
        elif path.is_file():
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"不支持的输入类型: {path}")
            files.append(path)
        else:
            raise FileNotFoundError(f"输入不存在: {path}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for item in files:
        resolved = item.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(item)

    if not unique:
        raise ValueError("没有找到可处理的 PDF 或图片")
    return unique


def _check_stem_collisions(files: list[Path]) -> None:
    seen: dict[str, Path] = {}
    for item in files:
        key = item.stem.lower()
        if key in seen:
            raise ValueError(
                f"输出文件名冲突: {seen[key]} 与 {item} 都叫 {item.stem}，"
                "请分批处理或重命名。"
            )
        seen[key] = item


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return str(resolved)
    return os.path.relpath(resolved, Path.cwd())


def _needs_org(fmt: str) -> bool:
    return fmt in ("org", "both")


def _already_done(fmt: str, md_path: Path, org_path: Path) -> bool:
    if fmt == "md":
        return md_path.is_file()
    if fmt == "org":
        return org_path.is_file()
    return md_path.is_file() and org_path.is_file()


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = _build_parser().parse_args(argv)

    try:
        files = _discover(args.inputs)
        _check_stem_collisions(files)
    except (ValueError, FileNotFoundError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve()
    try:
        check_ollama(args.server_url, args.model)
        if _needs_org(args.format) and not shutil.which("pandoc"):
            raise OcrError("找不到 pandoc，请先安装: brew install pandoc")
        ensure_layout_model(allow_download=not args.no_download)
    except OcrError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    succeeded = skipped = failed = 0

    for index, source in enumerate(files, start=1):
        md_path = out_dir / f"{source.stem}.md"
        org_path = out_dir / f"{source.stem}.org"
        print(f"[{index}/{len(files)}] {_display(source)}")

        if not args.force and _already_done(args.format, md_path, org_path):
            print("  跳过: 输出已存在（加 --force 覆盖）")
            skipped += 1
            continue

        work_md = md_path if args.format in ("md", "both") else out_dir / f".{source.stem}.md.partial"
        started = time.time()
        try:
            print("  识别中（版面检测 + OCR，可能需要几分钟）...")
            info = run_parse(
                source,
                work_md,
                server_url=args.server_url,
                model=args.model,
                workers=args.workers,
                timeout=args.timeout,
                keep_work=args.keep_work,
            )

            if _needs_org(args.format):
                org_text = md_to_org(work_md.read_text(encoding="utf-8"))
                org_path.write_text(org_text, encoding="utf-8")
                if args.format == "org":
                    work_md.unlink(missing_ok=True)

            elapsed = time.time() - started
            print(f"  完成: {info['pages'] or '?'} 页, {elapsed:.1f} 秒")
            if info.get("failed_blocks"):
                print(
                    f"  警告: {info['failed_blocks']}/{info['blocks']} 个块识别失败，结果可能不完整",
                    file=sys.stderr,
                )
            if args.format in ("md", "both"):
                print(f"  → {_display(md_path)}")
            if _needs_org(args.format):
                print(f"  → {_display(org_path)}")
            succeeded += 1
        except (OcrError, ConversionError, OSError, UnicodeDecodeError) as exc:
            print(f"  失败: {exc}", file=sys.stderr)
            if work_md.is_file():
                print(f"  中间 Markdown 保留在: {_display(work_md)}", file=sys.stderr)
            failed += 1
        except KeyboardInterrupt:
            print("\n已中断", file=sys.stderr)
            return 130

    print(f"\n汇总: 成功 {succeeded}, 跳过 {skipped}, 失败 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
