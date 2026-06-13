#!/usr/bin/env python3
"""
Export a Jupyter notebook to HTML via jupyter nbconvert.

Usage:
  python export_html.py
  python export_html.py 5004_HW3.ipynb
  python export_html.py HW3_solution.ipynb --output report.html
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def export_notebook_to_html(notebook_path: Path, output_name: str | None = None) -> None:
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    if notebook_path.suffix.lower() != ".ipynb":
        raise ValueError(f"Expected a .ipynb file, got: {notebook_path}")

    cmd = ["jupyter", "nbconvert", "--to", "html", str(notebook_path)]
    if output_name:
        cmd.extend(["--output", output_name])

    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Jupyter notebook to HTML.")
    parser.add_argument(
        "notebook",
        nargs="?",
        default="5004_HW3.ipynb",
        help="Path to notebook (.ipynb). Default: 5004_HW3.ipynb",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output HTML file name (e.g., result.html).",
    )
    args = parser.parse_args()

    notebook_path = Path(args.notebook).expanduser().resolve()

    try:
        export_notebook_to_html(notebook_path, args.output)
    except Exception as exc:  # pragma: no cover
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1

    print("Export completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
