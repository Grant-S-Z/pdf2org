"""Markdown to Org conversion for MiniCPM-OCR output."""

from __future__ import annotations

import re
import shutil
import subprocess


class ConversionError(RuntimeError):
    pass


_PAGE_BREAK_RE = re.compile(r"<!--\s*page-break\s*-->", re.IGNORECASE)
_EMPTY_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*\)")
_TABLE_TAG_RE = re.compile(r"(</?table\b[^>]*>)", re.IGNORECASE)
_DRAWER_RE = re.compile(r"\n:PROPERTIES:\n((?::[A-Za-z_]+:[^\n]*\n)+):END:\n")
_RULE_RE = re.compile(r"^-{5,}$", re.MULTILINE)
_HEADING_RE = re.compile(r"^(\*+ .+)\n(?=[^\s:])", re.MULTILINE)


def _drop_custom_id_drawer(match: re.Match[str]) -> str:
    if ":CUSTOM_ID:" in match.group(1):
        return "\n"
    return match.group(0)


def _preprocess(markdown: str) -> str:
    text = _PAGE_BREAK_RE.sub("\n\n-----\n\n", markdown)
    text = _EMPTY_IMAGE_RE.sub("", text)
    text = _TABLE_TAG_RE.sub(lambda m: "\n\n" + m.group(1) + "\n\n", text)
    return text


def _run_pandoc(args: list[str], text: str) -> str:
    try:
        proc = subprocess.run(
            args,
            input=text,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ConversionError(f"无法执行 pandoc: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or f"退出码 {proc.returncode}"
        raise ConversionError(f"pandoc 转换失败: {detail}")
    return proc.stdout


def md_to_org(markdown: str) -> str:
    if not shutil.which("pandoc"):
        raise ConversionError("找不到 pandoc，请先安装: brew install pandoc")

    html = _run_pandoc(
        [
            "pandoc",
            "-f",
            "markdown+tex_math_single_backslash+raw_html",
            "-t",
            "html5",
            "--wrap=none",
        ],
        _preprocess(markdown),
    )
    org = _run_pandoc(
        ["pandoc", "-f", "html", "-t", "org", "--wrap=none"],
        html,
    )

    org = _DRAWER_RE.sub(_drop_custom_id_drawer, org)
    lines = [line.rstrip() for line in org.splitlines()]
    org = "\n".join(lines)
    org = _RULE_RE.sub("-----", org)
    org = _HEADING_RE.sub(r"\1\n\n", org)
    org = re.sub(r"\n{3,}", "\n\n", org)
    return org.strip("\n") + "\n"
