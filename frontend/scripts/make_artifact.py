"""Turn the single-file demo build into a Claude artifact page (no <html>/<body>, title + meta first).

Usage: npm run build:demo && python frontend/scripts/make_artifact.py [out.html]
"""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = (root / "dist-demo" / "index.demo.html").read_text()
css = re.findall(r"<style[^>]*>(.*?)</style>", src, re.S)
js = re.findall(r'<script type="module"[^>]*>(.*?)</script>', src, re.S)
assert len(css) == 1 and len(js) == 1, "expected one inline style and one inline module script"
assert "</script" not in js[0].lower(), "the script contains a closing tag and cannot be inlined"

out = Path(sys.argv[1]) if len(sys.argv) > 1 else root.parent / "output" / "demo" / "labelhub-demo.html"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    '<title>LabelHub</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
    f'<style>{css[0]}</style>\n<div id="root"></div>\n<script type="module">{js[0]}</script>\n'
)
print(f"{out} ({out.stat().st_size / 1e6:.2f} MB)")
