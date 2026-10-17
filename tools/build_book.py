"""Build the GitBook pages (chapters/**.md and SUMMARY.md) from the slides and
the executed notebooks.

    python3 tools/build_book.py            # all chapters
    python3 tools/build_book.py ch1 ch2    # selected chapters (SUMMARY is rebuilt)

Slide frames are converted to markdown; right after the frames of an example
that cites a script (e.g. ``ch2_example_poisson.m``) the matching notebook is
inserted: its markdown cells, code, and outputs. A notebook split into parts
with ``# %% [part]`` markers is distributed over the successive groups of
frames that cite it.
"""
import base64
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import book_config as cfg  # noqa: E402
import percent  # noqa: E402
import tex2md as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT.parent
SLIDES = (ROOT / cfg.SLIDES_DIR).resolve()
ASSETS = ROOT / ".gitbook" / "assets"
FIG_DIRS = {
    "pathtofigs": CONTENT / "Matlab",
    "pathtodiags": CONTENT / "FigsBook",
    "pathtoexediags": CONTENT / "FigsExerciseBook",
    "pathtoexefigs": CONTENT / "MatlabExercises",
}
SRC, NB = ROOT / "src", ROOT / "notebooks"
WARN = []


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# ---------------------------------------------------------------- figures


def convert_image(macro, name):
    base = FIG_DIRS.get(macro)
    if base is None:
        WARN.append(f"unknown figure path macro {macro}")
        return None
    name = name.strip()
    stem = re.sub(r"\.(eps|pdf|png|jpe?g)$", "", Path(name).name)
    cands = [base / name] + [base / (name + ext) for ext in (".eps", ".pdf", ".png", ".jpg", ".jpeg")]
    src = next((c for c in cands if c.is_file()), None)
    if src is None:  # fall back on the pdf that epstopdf produced for the slides
        alt = SLIDES / "FigsOutput" / (stem + "-eps-converted-to.pdf")
        src = alt if alt.is_file() else None
    if src is None:
        WARN.append(f"missing figure {macro}/{name}")
        return None
    out_dir = ASSETS / "slides"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    out = out_dir / (stem + (".jpg" if ext in (".jpg", ".jpeg") else ".png"))
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    if ext in (".jpg", ".jpeg", ".png"):
        out.write_bytes(src.read_bytes())
    else:
        crop = ["-dEPSCrop"] if ext == ".eps" else ["-dUseCropBox"]
        cmd = ["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r200",
               "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", "-dFirstPage=1", "-dLastPage=1",
               *crop, f"-sOutputFile={out}", str(src)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode or not out.exists():
            WARN.append(f"ghostscript failed for {src}: {r.stderr[:200]}")
            return None
    return out


# ---------------------------------------------------------------- slide parsing


class Ctx:
    def __init__(self, page_dir):
        self.page_dir = page_dir
        self.scripts = []

    def script_name(self, url):
        m = re.search(r"(ch\d_[A-Za-z0-9_]+)\.m\b", url)
        return m.group(1) if m else None

    def href(self, url, txt):
        name = self.script_name(url)
        if name:
            if name not in self.scripts:
                self.scripts.append(name)
            # drop the "ch2_example.m" label; keep figures wrapped in the link
            return "" if re.fullmatch(r"\s*`?[\w.\\]*`?\s*", T.text_cmds(txt, self)) else txt
        url = url.replace("\\pathtoscripts", "https://github.com/vzavala/StatsBook/tree/master/Matlab/")
        return f"[{txt}]({url.strip()})"

    def rel(self, p):
        return Path("../" * len(self.page_dir.relative_to(ROOT).parts)) / p.relative_to(ROOT)


def figure_html(ctx, imgs, caption):
    parts = []
    for macro, name, width in imgs:
        out = convert_image(macro, name)
        if out is None:
            continue
        w = int(min(width, 1.0) * 700)
        parts.append(f'<img src="{ctx.rel(out)}" alt="" width="{w}">')
    if not parts:
        return ""
    # GitBook figures hold a single image: one <figure> per image, caption on the last.
    # GitBook does not render math inside HTML, so a caption with math goes in an
    # italic Markdown line right below the image instead of in <figcaption>.
    caption = caption.strip()
    math_caption = "$$" in caption
    cap = f"<figcaption><p>{caption}</p></figcaption>" if caption and not math_caption else ""
    figs = [f"<figure>{img}{cap if k == len(parts) - 1 else ''}</figure>" for k, img in enumerate(parts)]
    if math_caption:
        figs[-1] += f"\n\n*{caption}*"
    return "\n\n" + "\n\n".join(figs) + "\n\n"


GRAPHIC = re.compile(r"\\includegraphics\s*(\[[^\]]*\])?\s*\{\s*\\(path[a-z]+)\s*([^}]*)\}")


def graphics_in(s):
    out = []
    for m in GRAPHIC.finditer(s):
        opt = m.group(1) or ""
        w = re.search(r"width\s*=\s*([\d.]*)\s*\\(text|line|column)width", opt)
        width = float(w.group(1) or 1) if w else 0.7
        out.append((m.group(2), m.group(3), width))
    return out


def convert_body(body, ctx, macros):
    store = T.Store()
    s = T.expand_macros(body, macros)
    s = T.extract_math(s, store)
    s = T.replace_cmd(s, "caption", lambda t: "\x02CAP" + t + "\x03")

    def fig_env(m):
        inner = m.group(1)
        cap = re.search(r"\x02CAP(.*?)\x03", inner, re.S)
        caption = T.text_cmds(cap.group(1), ctx).strip() if cap else ""
        # keep \href{script}{...} wrappers so the script is recorded
        for h in re.finditer(r"\\href\{([^}]*)\}", inner):
            ctx.href(h.group(1), "")
        return store.put(figure_html(ctx, graphics_in(inner), store.restore(caption)))
    s = re.sub(r"\\begin\{figure\*?\}(?:\[[^\]]*\])?(.*?)\\end\{figure\*?\}", fig_env, s, flags=re.S)
    s = re.sub(r"\\begin\{table\}(?:\[[^\]]*\])?|\\end\{table\}", "", s)
    s = GRAPHIC.sub(lambda m: store.put(figure_html(ctx, graphics_in(m.group(0)), "")), s)
    s = re.sub(r"\x02CAP(.*?)\x03", lambda m: "\n\n*" + m.group(1).strip() + "*\n\n", s, flags=re.S)

    def tab(m):
        return store.put(T.convert_tabular(m.group(2), ctx))
    s = re.sub(r"\\begin\{tabular\}\s*(\{(?:[^{}]|\{[^{}]*\})*\})(.*?)\\end\{tabular\}", tab, s, flags=re.S)
    s = re.sub(r"\\begin\{column\}\s*(\[[^\]]*\])?\s*\{[^}]*\}|\\end\{column\}", "\n", s)
    s = re.sub(r"\\begin\{minipage\}\s*(\[[^\]]*\])?\s*\{[^}]*\}|\\end\{minipage\}", "\n", s)
    s = re.sub(r"\\(begin|end)\{(columns|center|footnotesize|small|tiny|flushleft|algorithmic)\}(\[[^\]]*\])?",
               "\n", s)
    md = T.convert_lists(s, ctx)
    return T.tidy(store.restore(md))


def parse_tex(path):
    tex = path.read_text(encoding="utf-8", errors="replace")
    pre, _, body = tex.partition("\\begin{document}")
    macros = T.parse_macros(T.strip_comments(pre))
    body = T.strip_comments(body.split("\\end{document}")[0])
    frames = []
    for m in re.finditer(r"\\begin\{frame\}", body):
        i = m.end()
        j = T.skip_ws(body, i)
        if j < len(body) and body[j] == "[":
            _, i = T.read_group(body, j, "[", "]")
            j = T.skip_ws(body, i)
        title = ""
        if j < len(body) and body[j] == "{":
            title, i = T.read_group(body, j)
        end = body.find("\\end{frame}", i)
        content = body[i:end]
        if "\\titlepage" in content:
            continue
        frames.append(dict(title_tex=title, body=content, macros=macros))
    return frames


def frame_title(title_tex, ctx, macros):
    store = T.Store()
    s = T.extract_math(T.expand_macros(title_tex, macros), store)
    s = T.text_cmds(s, ctx)
    s = store.restore(s)
    # script names written without \href: "`ch2_example_uniform.m`", "\tt example_bernoulli"
    for m in re.finditer(r"`?(ch\d_[A-Za-z0-9_]+)\.m`?", s):
        if m.group(1) not in ctx.scripts:
            ctx.scripts.append(m.group(1))
    s = re.sub(r"`?ch\d_[A-Za-z0-9_]+(\.m)?`?|\\tt\s+\S+", "", s)
    return re.sub(r"\s+", " ", s).strip(" .")


# ---------------------------------------------------------------- notebooks


def colab_url(nb):
    return (f"https://colab.research.google.com/github/{cfg.GITHUB_REPO}/blob/"
            f"{cfg.GITHUB_BRANCH}/notebooks/{nb}.ipynb")


def nb_math(md):
    """Jupyter $x$ -> GitBook $$x$$; display $$..$$ on its own lines."""
    store = T.Store()
    md = re.sub(r"\$\$(.+?)\$\$", lambda m: store.put("\n$$\n" + m.group(1).strip() + "\n$$\n"), md, flags=re.S)
    md = re.sub(r"(?<![\\$])\$([^$\n]+?)\$", lambda m: store.put("$$" + T.bracket_slices(m.group(1)) + "$$"), md)
    md = re.sub(r"\\begin\{(align\*?|equation\*?)\}(.*?)\\end\{\1\}",
                lambda m: store.put("\n$$\n\\begin{aligned}" + m.group(2) + "\\end{aligned}\n$$\n"),
                md, flags=re.S)
    return store.restore(md)


def load_cells(name):
    """Cells with outputs from notebooks/<name>.ipynb (fallback: src without outputs)."""
    ipynb, src = NB / f"{name}.ipynb", SRC / f"{name}.py"
    if ipynb.exists() and (not src.exists() or ipynb.stat().st_mtime >= src.stat().st_mtime):
        nb = json.loads(ipynb.read_text())
        cells = []
        for c in nb["cells"]:
            meta = c.get("metadata", {})
            cells.append(dict(type=c["cell_type"], source="".join(c["source"]),
                              hide="book-hide" in meta.get("tags", []),
                              part=meta.get("part", 0), outputs=c.get("outputs", []),
                              is_part="part_title" in meta, part_title=meta.get("part_title")))
        return cells
    if src.exists():
        WARN.append(f"{name}: no executed notebook, rendering without outputs")
        return [dict(type=c["type"], source=c["source"], hide=c["hide"], part=c["part"],
                     outputs=[], is_part=c["type"] == "part",
                     part_title=c["source"] if c["type"] == "part" else None) for c in percent.parse(src.read_text())]
    WARN.append(f"no notebook for {name}")
    return None


def render_outputs(outputs, name, idx, ctx):
    md = []
    for k, o in enumerate(outputs):
        data = o.get("data", {})
        if o.get("output_type") == "stream" and o.get("name") == "stdout":
            txt = "".join(o.get("text", "")).rstrip()
            if txt:
                md.append(_text_block(txt))
        elif "image/png" in data:
            out = ASSETS / "nb" / f"{name}-{idx:02d}-{k}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(base64.b64decode(data["image/png"]))
            md.append(f'\n<figure><img src="{ctx.rel(out)}" alt="" width="520"></figure>\n')
        elif "text/plain" in data and o.get("output_type") == "execute_result":
            txt = "".join(data["text/plain"]).rstrip()
            if txt and not re.fullmatch(r"<.*>", txt) and not txt.startswith(("Text(", "[<matplotlib", "(np.float")):
                md.append(_text_block(txt))
    return "\n".join(md)


def _text_block(txt, limit=30):
    lines = txt.splitlines()
    if len(lines) > limit:
        lines = lines[:limit] + ["..."]
    return "\n```\n" + "\n".join(lines) + "\n```\n"


def render_notebook(name, cells, ctx, first, part_title=None):
    collapsed = name.startswith(cfg.COLLAPSED_PREFIXES)
    body = []
    for idx, c in cells:
        if c["hide"]:
            continue
        if c.get("is_part"):
            continue
        if c["type"] == "markdown":
            body.append(nb_math(c["source"]) + "\n")
        elif c["type"] == "code":
            body.append("```python\n" + c["source"].rstrip() + "\n```\n")
            body.append(render_outputs(c.get("outputs", []), name, idx, ctx))
    if not any(b.strip() for b in body):
        return ""
    label = f"[`{name}.ipynb`]({colab_url(name)})"
    if collapsed:
        return ("\n<details>\n\n<summary>Python: reproduce this figure (" + name + ".ipynb)</summary>\n\n"
                + "\n".join(body) + f"\n\nOpen in Colab: {label}\n\n</details>\n\n")
    head = "#### Python implementation" + (f": {part_title}" if part_title else "")
    badge = (f"{{% hint style=\"success\" %}}\nNotebook {label} — open it in Colab and run the cells "
             f"alongside this section.\n{{% endhint %}}\n\n") if first else ""
    return f"\n{head}\n\n{badge}" + "\n".join(body) + "\n"


def notebook_chunks(name):
    """Split cells into parts (part 0 is merged into part 1)."""
    cells = load_cells(name)
    if cells is None:
        return []
    chunks, titles = {}, {}
    for idx, c in enumerate(cells):
        p = max(c["part"], 1)
        chunks.setdefault(p, []).append((idx, c))
        if c.get("is_part"):
            titles[p] = (c.get("part_title") or c["source"].lstrip("# ")).strip() or None
    return [(titles.get(p), chunks[p]) for p in sorted(chunks)]


def split_anchor(title):
    """'Quantiles @ Summarizing Statistics (Quantiles)' -> ('Quantiles', 'Summarizing...')."""
    if title and " @ " in title:
        t, a = title.split(" @ ", 1)
        return t.strip() or None, a.strip()
    return title, None


# ---------------------------------------------------------------- pages


def title_runs(frames, prefix):
    """Last frame index of each run of consecutive frames whose title starts with prefix."""
    runs = []
    for i, f in enumerate(frames):
        if f["title"].startswith(prefix):
            if runs and runs[-1] == i - 1 and frames[i - 1]["title"] == f["title"]:
                runs[-1] = i
            else:
                runs.append(i)
    return runs


def build_chapter(ch):
    tex = SLIDES / ch["tex"]
    frames = parse_tex(tex)
    folder = ROOT / "chapters" / ch["key"]
    folder.mkdir(parents=True, exist_ok=True)

    # convert frames
    for f in frames:
        ctx = Ctx(folder)
        f["title"] = frame_title(f["title_tex"], ctx, f["macros"])
        f["md"] = convert_body(f["body"], ctx, f["macros"])
        f["scripts"] = [cfg.ALIASES.get(s, s) for s in ctx.scripts]

    # extra notebooks attached to a frame by title
    for nb, (key, prefix, occ) in cfg.EXTRA.items():
        if key != ch["key"]:
            continue
        runs = title_runs(frames, prefix)
        if occ < 0:
            occ = len(runs) + 1 + occ
        if not runs or not 1 <= occ <= len(runs):
            WARN.append(f"EXTRA {nb}: frame '{prefix}' #{occ} not found in {key}")
            continue
        frames[runs[occ - 1]]["scripts"].append(nb)

    # page boundaries
    starts, pos = [], 0
    for title, prefix in ch["pages"]:
        k = next((i for i in range(pos, len(frames)) if frames[i]["title"].startswith(prefix)), None)
        if k is None:
            WARN.append(f"{ch['key']}: page start '{prefix}' not found")
            continue
        starts.append((title, k))
        pos = k + 1
    if starts and starts[0][1] != 0:
        starts[0] = (starts[0][0], 0)

    # notebook placement: groups of consecutive frames citing a notebook
    inserts = {}  # frame index -> list of (name, part index, first?)
    names = []
    for i, f in enumerate(frames):
        for s in f["scripts"]:
            if s not in names:
                names.append(s)
    for name in names:
        idxs = [i for i, f in enumerate(frames) if name in f["scripts"]]
        groups = []
        for i in idxs:
            if groups and i - groups[-1][-1] <= 1:
                groups[-1].append(i)
            else:
                groups.append([i])
        chunks = notebook_chunks(name)
        if not chunks:
            continue
        g_next = 0
        for p, (ptitle, cells) in enumerate(chunks):
            ptitle, anchor = split_anchor(ptitle)
            if anchor:
                occ = -1                       # "Title #k": k-th frame with that title
                m = re.match(r"(.*?)\s+#(\d+)$", anchor)
                if m:
                    anchor, occ = m.group(1), int(m.group(2)) - 1
                hits = title_runs(frames, anchor)
                if not hits or occ >= len(hits):
                    WARN.append(f"{name}: anchor frame '{anchor}' not found")
                    continue
                at = hits[occ]
            elif groups and len(chunks) == 1:
                at = groups[-1][-1]            # single chunk: after the last citing frames
            elif groups:
                at = groups[min(g_next, len(groups) - 1)][-1]
                g_next += 1
            else:
                continue
            inserts.setdefault(at, []).append((name, ptitle, cells, p == 0))

    pages = []
    for n, (title, k) in enumerate(starts):
        end = starts[n + 1][1] if n + 1 < len(starts) else len(frames)
        ctx = Ctx(folder)
        md = [f"# {title}\n"]
        prev = None
        for i in range(k, end):
            f = frames[i]
            if f["title"] and f["title"] != prev and not (i == k and f["title"] == title):
                md.append(f"\n## {f['title']}\n")
            prev = f["title"] or prev
            md.append(f["md"])
            for name, ptitle, cells, first in inserts.get(i, []):
                md.append(render_notebook(name, cells, ctx, first, ptitle if len(notebook_chunks(name)) > 1 else None))
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(md)).strip() + "\n"
        fname = f"{n + 1:02d}-{slug(title)}.md"
        (folder / fname).write_text(text)
        pages.append((title, f"chapters/{ch['key']}/{fname}"))
    # drop stale pages
    keep = {Path(p).name for _, p in pages}
    for old in folder.glob("*.md"):
        if old.name not in keep and old.name != "README.md":
            old.unlink()
    return pages


def write_summary(all_pages):
    lines = ["# Table of contents\n", "* [Statistics for Chemical Engineers](README.md)",
             "* [How to use this book](how-to-use.md)",
             "* [Nomenclature: math and Python](nomenclature.md)", ""]
    current_group = None
    for ch, pages in all_pages:
        group = ch.get("group")
        if group:
            if group != current_group:
                lines += ["", f"## {group}", ""]
                current_group = group
            first_title, first_path = pages[0]
            lines.append(f"* [{ch['title']}]({first_path})")
            lines += [f"  * [{t}]({p})" for t, p in pages[1:]]
        else:
            current_group = None
            lines += ["", f"## {ch['title']}", ""]
            lines += [f"* [{t}]({p})" for t, p in pages]
    (ROOT / "SUMMARY.md").write_text("\n".join(lines).replace("\n\n\n", "\n\n") + "\n")


def write_nomenclature(all_pages):
    """nomenclature.md = conventions (tools/nomenclature_intro.md) + each notebook's symbol table."""
    out = [(Path(__file__).parent / "nomenclature_intro.md").read_text().rstrip() + "\n"]
    seen = set()
    for ch, pages in all_pages:
        section = []
        for _, page in pages:
            for name in re.findall(r"notebooks/(ch\d_[A-Za-z0-9_]+)\.ipynb", (ROOT / page).read_text()):
                if name in seen or not (SRC / f"{name}.py").exists():
                    continue
                seen.add(name)
                cells = percent.parse((SRC / f"{name}.py").read_text())
                table = next((c["source"] for c in cells if c["type"] == "markdown"
                              and "| Slides | Python |" in c["source"]), None)
                if table is None:
                    continue
                rows = [l for l in table.splitlines() if l.startswith("|")]
                section.append(f"#### [`{name}`]({page})\n\n" + nb_math("\n".join(rows)) + "\n")
        if section:
            title = ch.get("group", "") + (" — " if ch.get("group") else "") + ch["title"]
            out.append(f"\n### {title}\n\n" + "\n".join(section))
    (ROOT / "nomenclature.md").write_text("\n".join(out))


def main():
    want = set(sys.argv[1:])
    all_pages = []
    for ch in cfg.CHAPTERS:
        if not want or ch["key"] in want:
            pages = build_chapter(ch)
            print(f"{ch['key']}: {len(pages)} pages")
        else:  # keep existing pages in the summary
            folder = ROOT / "chapters" / ch["key"]
            pages = [(t, f"chapters/{ch['key']}/{n + 1:02d}-{slug(t)}.md") for n, (t, _) in enumerate(ch["pages"])
                     if (folder / f"{n + 1:02d}-{slug(t)}.md").exists()]
        if pages:
            all_pages.append((ch, pages))
    write_summary(all_pages)
    write_nomenclature(all_pages)
    for w in WARN:
        print("warning:", w)


if __name__ == "__main__":
    main()
