#!/usr/bin/env python3
"""Convert a Jupyter Notebook (.ipynb) to HTML using nbconvert API.

Usage:
    python3 scripts/convert_notebook_to_html.py 5053_HW3.ipynb output.html
If output filename is omitted, it will use the notebook basename with .html.
"""
import sys
import nbformat
from nbconvert import HTMLExporter
from traitlets.config import Config


def convert_notebook(input_path: str, output_path: str) -> None:
    nb = nbformat.read(input_path, as_version=4)
    c = Config()
    # Tweak exporter settings if desired
    c.HTMLExporter.exclude_output_prompt = True
    c.HTMLExporter.exclude_input = False
    exporter = HTMLExporter(config=c)
    body, resources = exporter.from_notebook_node(nb)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/convert_notebook_to_html.py input.ipynb [output.html]")
        sys.exit(1)
    input_nb = sys.argv[1]
    if len(sys.argv) >= 3:
        output_html = sys.argv[2]
    else:
        output_html = input_nb.rsplit('.', 1)[0] + '.html'
    convert_notebook(input_nb, output_html)
