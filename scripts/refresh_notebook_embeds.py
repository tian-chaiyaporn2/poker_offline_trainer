#!/usr/bin/env python3
"""Regenerate base64-embedded source ZIPs inside Colab/Kaggle notebooks.

Keeps notebook embeds in sync with src/ so validation notebooks do not run
stale solver code. Usage:
    python3 scripts/refresh_notebook_embeds.py
"""

from __future__ import annotations

import base64
import io
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "colab" / "poker_fullrange_validation.ipynb",
    ROOT / "colab" / "kaggle_fullrange_validation.ipynb",
    ROOT / "colab" / "kaggle_content_yield.ipynb",
    ROOT / "colab" / "poker_gpu_benchmark_selfcontained.ipynb",
]


def build_zip_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for base in (ROOT / "src" / "pokertrainer", ROOT / "bench"):
            for path in sorted(base.rglob("*")):
                if not path.is_file():
                    continue
                if "__pycache__" in path.parts:
                    continue
                if path.suffix not in {".py", ".c"}:
                    continue
                z.write(path, path.relative_to(ROOT).as_posix())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def format_b64_literal(b64: str, width: int = 76) -> str:
    chunks = [b64[i:i + width] for i in range(0, len(b64), width)]
    lines = ["_ZIP_B64 = ("]
    for c in chunks:
        lines.append(f'    "{c}"')
    lines.append(")")
    return "\n".join(lines)


def refresh_notebook(path: Path, literal: str) -> bool:
    nb = json.loads(path.read_text())
    changed = False
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if "_ZIP_B64" not in src:
            continue
        new_src, n = re.subn(
            r"_ZIP_B64\s*=\s*\(.*?\)\n",
            literal + "\n",
            src,
            count=1,
            flags=re.DOTALL,
        )
        if n != 1:
            raise SystemExit(f"failed to replace _ZIP_B64 in {path}")
        # Jupyter source is a list of lines ending with \n (except possibly last)
        lines = new_src.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        cell["source"] = lines
        changed = True
        break
    if changed:
        path.write_text(json.dumps(nb, indent=1) + "\n")
    return changed


def main() -> None:
    literal = format_b64_literal(build_zip_b64())
    for nb in NOTEBOOKS:
        if not nb.exists():
            print(f"skip missing {nb}")
            continue
        if refresh_notebook(nb, literal):
            print(f"updated {nb.relative_to(ROOT)}")
        else:
            print(f"no embed in {nb.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
