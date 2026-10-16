"""Approximate local preview of the GitBook (math via KaTeX, hints, figures).

    python tools/preview.py [outdir]     # needs the `markdown` package

Writes one HTML file per page (default: _preview/), mirroring SUMMARY.md, so the
pages can be checked in a browser before pushing to GitBook. It is only an
approximation of GitBook's renderer.
"""
import html
import shutil
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_preview"

HEAD = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[{{left:'\\\\[',right:'\\\\]',display:true}},
 {{left:'\\\\(',right:'\\\\)',display:false}}],throwOnError:false}})"></script>
<style>
body{{font:16px/1.6 -apple-system,Segoe UI,sans-serif;margin:0;display:flex;color:#1d1d1f;background:#fff}}
nav{{width:280px;padding:16px;border-right:1px solid #ddd;height:100vh;overflow:auto;position:sticky;top:0;font-size:14px}}
nav a{{display:block;color:#333;text-decoration:none;padding:2px 0}} nav h4{{margin:14px 0 4px}}
main{{max-width:820px;padding:24px 40px;overflow-x:hidden}}
pre{{background:#f6f8fa;padding:12px;overflow:auto;border-radius:6px;font-size:13px}}
.hint{{border-left:4px solid #4a90e2;background:#f1f6fd;padding:8px 14px;margin:12px 0;border-radius:4px}}
.hint.success{{border-color:#2e9e5b;background:#eefaf2}}
figure{{margin:12px 0}} img{{max-width:100%}} figcaption{{font-size:13px;color:#555}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 8px}}
.katex-error{{background:#fdd}}
</style></head><body><nav>{nav}</nav><main>{body}</main></body></html>"""


def protect_math(md):
    store = []

    def put(s):
        store.append(s)
        return f"MATHTOKEN{len(store) - 1}END"
    # display blocks: $$ on its own line (possibly indented)
    md = re.sub(r"(?m)^([ \t]*)\$\$\s*\n(.*?)\n[ \t]*\$\$[ \t]*$",
                lambda m: m.group(1) + put("\\[" + html.escape(m.group(2)) + "\\]"), md, flags=re.S)
    md = re.sub(r"\$\$(.+?)\$\$", lambda m: put("\\(" + html.escape(m.group(1)) + "\\)"), md)
    return md, store


def render(md, store):
    md = re.sub(r'\{% hint style="(\w+)" %\}', lambda m: f'<div class="hint {m.group(1)}" markdown="1">', md)
    md = md.replace("{% endhint %}", "</div>")
    out = markdown.markdown(md, extensions=["fenced_code", "tables", "md_in_html", "sane_lists"])
    return re.sub(r"MATHTOKEN(\d+)END", lambda m: store[int(m.group(1))], out)


def main():
    summary = (ROOT / "SUMMARY.md").read_text()
    pages = re.findall(r"\[([^\]]+)\]\(([^)]+\.md)\)", summary)
    nav = []
    for line in summary.splitlines():
        m = re.match(r"\s*\* \[([^\]]+)\]\(([^)]+)\.md\)", line)
        if m:
            nav.append(f'<a href="/{m.group(2)}.html">{html.escape(m.group(1))}</a>')
        elif line.startswith("## "):
            nav.append(f"<h4>{html.escape(line[3:])}</h4>")
    nav = "\n".join(nav)
    for title, rel in pages:
        src = ROOT / rel
        if not src.exists():
            continue
        md, store = protect_math(src.read_text())
        body = render(md, store)
        # assets are referenced relative to the page; keep paths valid from OUT
        dst = OUT / Path(rel).with_suffix(".html")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(HEAD.format(title=html.escape(title), nav=nav, body=body))
    assets = OUT / ".gitbook"
    if assets.is_symlink():
        assets.unlink()
    shutil.copytree(ROOT / ".gitbook", assets, dirs_exist_ok=True)
    print(f"wrote {len(pages)} pages to {OUT}")


if __name__ == "__main__":
    main()
