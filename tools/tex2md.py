"""Convert the beamer slide sources (Slides/Slides-New/*.tex) to GitBook markdown.

Only the LaTeX subset used by the slides is handled: frames, sections, itemize /
enumerate, blocks, figures, columns, tabular, inline and display math.
Math is emitted in GitBook/KaTeX form: inline `$$...$$`, display on own lines.
"""
import re

# ---------------------------------------------------------------- helpers


def read_group(s, i, open_="{", close="}"):
    """s[i] == open_; return (content, index after the matching close)."""
    assert s[i] == open_, s[i:i + 30]
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced group: " + s[i:i + 60])


def skip_ws(s, i):
    while i < len(s) and s[i] in " \t\n":
        i += 1
    return i


def strip_comments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in s.splitlines())


def replace_cmd(s, name, fn, nargs=1, optional=False):
    """Replace \\name[opt]{a1}..{an} by fn(*args) (balanced braces)."""
    pat = re.compile(r"\\" + name + r"(?![a-zA-Z])")
    out, pos = [], 0
    while True:
        m = pat.search(s, pos)
        if not m:
            break
        i = m.end()
        opt = None
        j = skip_ws(s, i)
        if optional and j < len(s) and s[j] == "[":
            opt, i = read_group(s, j, "[", "]")
        args, ok = [], True
        for _ in range(nargs):
            j = skip_ws(s, i)
            if j >= len(s) or s[j] != "{":
                ok = False
                break
            a, i = read_group(s, j)
            args.append(a)
        if not ok:
            out.append(s[pos:m.end()])
            pos = m.end()
            continue
        out.append(s[pos:m.start()])
        out.append(fn(*args, opt=opt) if optional else fn(*args))
        pos = i
    out.append(s[pos:])
    return "".join(out)


def replace_group_cmd(s, name, fn):
    """Replace {\\name text} (declaration form) by fn(text)."""
    pat = re.compile(r"\{\s*\\" + name + r"(?![a-zA-Z])")
    while True:
        m = pat.search(s)
        if not m:
            return s
        body, end = read_group(s, m.start())
        inner = re.sub(r"^\s*\\" + name + r"(?![a-zA-Z])", "", body, count=1)
        s = s[:m.start()] + fn(inner) + s[end:]


# ---------------------------------------------------------------- macros

MATH_OVERRIDES = {
    r"\st": r"\text{s.t.}",
    r"\rank": r"\operatorname{rank}",
    r"\diag": r"\operatorname{diag}",
}


def parse_macros(preamble):
    macros = {}
    for m in re.finditer(r"\\newcommand\{?(\\[a-zA-Z]+)\}?\{", preamble):
        name = m.group(1)
        try:
            body, _ = read_group(preamble, m.end() - 1)
        except ValueError:
            continue
        if "#" in body or name.startswith(r"\path") or name in (r"\blfootnote",):
            continue
        macros[name] = body
    macros.update(MATH_OVERRIDES)
    for k in (r"\red", r"\blue", r"\green", r"\UWgray", r"\UWred", r"\itsepbig",
              r"\itsepsm", r"\itsepBig", r"\backupbegin", r"\backupend"):
        macros[k] = ""
    return macros


def expand_macros(s, macros):
    for name in sorted(macros, key=len, reverse=True):
        s = re.sub(re.escape(name) + r"(?![a-zA-Z])", lambda _m, b=macros[name]: b, s)
    return s


# ---------------------------------------------------------------- math

DISPLAY_ENVS = ["align*", "align", "equation*", "equation", "gather*", "gather",
                "eqnarray*", "eqnarray", "multline*"]


SLIDE_TYPOS = {r"\matcal{": r"\mathcal{", r"\kapp\,": r"\kappa\,"}


def tabular_to_array(m):
    """KaTeX has no tabular: use array, and drop the $...$ inside cells."""
    def rep(t):
        return "\\begin{array}" + t.group(1) + t.group(2).replace("$", "") + "\\end{array}"
    return re.sub(r"\\begin\{tabular\}(\{[^}]*\})(.*?)\\end\{tabular\}", rep, m, flags=re.S)


def siunitx_num(m):
    """\\num{2.44e-3} -> 2.44\\times 10^{-3}"""
    def rep(t):
        v = t.group(1).strip()
        mm = re.fullmatch(r"([-+]?[\d.]+)[eE]([-+]?\d+)", v)
        if not mm:
            return v
        return f"{mm.group(1)}\\times 10^{{{int(mm.group(2))}}}"
    return re.sub(r"\\num\{([^}]*)\}", rep, m)


def bracket_slices(m):
    """X[:,1] -> X\\lbrack :,1\\rbrack (same rendering; GitBook's parser trips on [:...] in math)."""
    return re.sub(r"\[(:[^\[\]]*)\]", r"\\lbrack \1\\rbrack ", m)


def clean_math(m):
    for bad, good in SLIDE_TYPOS.items():
        m = m.replace(bad, good)
    m = siunitx_num(m)
    m = tabular_to_array(m)
    m = bracket_slices(m)
    m = re.sub(r"\\label\{[^}]*\}", "", m)
    m = re.sub(r"\\(nonumber|notag)\b", "", m)
    m = re.sub(r"\\(displaystyle)\b", r"\\displaystyle", m)
    m = re.sub(r"\\textup\{", r"\\textrm{", m)
    m = re.sub(r"\\mathbf\{\\(beta|epsilon|lambda|Lambda|Sigma|xi|theta|mu|phi|Phi)\}",
               r"\\boldsymbol{\\\1}", m)
    m = re.sub(r"\\bf\b", r"\\mathbf", m)
    m = re.sub(r"\\vspace\{[^}]*\}", "", m)
    m = re.sub(r"\\(color)\{[^}]*\}", "", m)
    m = m.replace(r"\\[2pt]", r"\\").replace("\\\\[5pt]", "\\\\")
    m = re.sub(r"\n\s*\n", "\n", m)
    return m.strip()


class Store:
    """Placeholder store so math/code survives text processing."""

    def __init__(self):
        self.items = []

    def put(self, text):
        self.items.append(text)
        return f"\x00{len(self.items) - 1}\x01"

    def restore(self, s):
        for _ in range(4):
            s = re.sub(r"\x00(\d+)\x01", lambda m: self.items[int(m.group(1))], s)
        return s


def display_block(body):
    return "\n\n$$\n" + body + "\n$$\n\n"


def extract_math(s, store):
    s = s.replace("$``$", "\u201c").replace("$\"$", "\u201d").replace("$''$", "\u201d")
    s = re.sub(r"\\begin\{subequations\}|\\end\{subequations\}", "", s)
    for env in DISPLAY_ENVS:
        # tolerate a starred \begin closed by an unstarred \end (and vice versa)
        e = re.escape(env.rstrip("*")) + r"\*?"
        aligned = env.startswith(("align", "eqnarray", "gather"))

        def rep(m, aligned=aligned):
            body = clean_math(m.group(1))
            if aligned and "&" in body or aligned and "\\\\" in body:
                body = "\\begin{aligned}\n" + body + "\n\\end{aligned}"
            return store.put(display_block(body))
        s = re.sub(r"\\begin\{" + e + r"\}(.*?)\\end\{" + e + r"\}", rep, s, flags=re.S)
    s = re.sub(r"\\\[(.*?)\\\]", lambda m: store.put(display_block(clean_math(m.group(1)))), s, flags=re.S)
    s = re.sub(r"\$\$(.*?)\$\$", lambda m: store.put(display_block(clean_math(m.group(1)))), s, flags=re.S)
    s = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$",
               lambda m: store.put("$$" + clean_math(m.group(1)).replace("\n", " ") + "$$"),
               s, flags=re.S)
    return s


# ---------------------------------------------------------------- text


def text_cmds(s, ctx):
    s = replace_cmd(s, "href", lambda url, txt: ctx.href(url, txt), 2)
    s = replace_cmd(s, "url", lambda url: f"<{url}>")
    s = replace_group_cmd(s, "em", lambda t: f"*{t.strip()}*")
    s = replace_group_cmd(s, "it", lambda t: f"*{t.strip()}*")
    s = replace_group_cmd(s, "bf", lambda t: f"**{t.strip()}**")
    s = replace_group_cmd(s, "tt", lambda t: f"`{t.strip()}`")
    for size in ("footnotesize", "small", "tiny", "large", "Large", "LARGE", "normalsize",
                 "scriptsize", "huge"):
        s = replace_group_cmd(s, size, lambda t: t)
    s = replace_cmd(s, "textbf", lambda t: f"**{t.strip()}**")
    s = replace_cmd(s, "emph", lambda t: f"*{t.strip()}*")
    s = replace_cmd(s, "textit", lambda t: f"*{t.strip()}*")
    s = replace_cmd(s, "underline", lambda t: f"<u>{t}</u>")
    s = replace_cmd(s, "texttt", lambda t: "`" + t.replace("\\_", "_") + "`")
    s = replace_cmd(s, "textrm", lambda t: t)
    s = replace_cmd(s, "footnote", lambda t: f" ({t})")
    s = replace_cmd(s, "blfootnote", lambda t: f"\n\n<sub>{t}</sub>\n\n")
    s = replace_cmd(s, "vspace", lambda t: "", 1)
    s = replace_cmd(s, "hspace", lambda t: "", 1)
    s = replace_cmd(s, "setlength", lambda a, b: "", 2)
    s = replace_cmd(s, "setcounter", lambda a, b: "", 2)
    s = replace_cmd(s, "addtocounter", lambda a, b: "", 2)
    s = replace_cmd(s, "label", lambda t: "")
    s = replace_cmd(s, "color", lambda t: "")
    s = replace_cmd(s, "cite", lambda t: "")
    s = replace_cmd(s, "ref", lambda t: "above")
    s = replace_cmd(s, "eqref", lambda t: "")
    s = re.sub(r"\\(centering|medskip|bigskip|smallskip|noindent|par|pause|newline|hfill|"
               r"footnotesize|small|tiny|large|Large|normalsize|scriptsize|em|bf|it|tt|rm|sf)(?![a-zA-Z])\s*",
               "", s)
    s = re.sub(r"\\itsep[a-zA-Z]*", "", s)
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    s = s.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_").replace("\\#", "#")
    s = s.replace("\\$", "$").replace("~", " ")
    s = s.replace("\\copyright", "\u00a9").replace("\\ldots", "...").replace("\\dots", "...")
    s = re.sub(r"\\\\(\[[^\]]*\])?", "\n", s)
    s = re.sub(r"(?<!\\)[{}]", "", s)
    return s


# ---------------------------------------------------------------- structure


def convert_tabular(body, ctx):
    body = re.sub(r"\\(hline|toprule|midrule|bottomrule)", "", body)
    rows = [r.strip() for r in re.split(r"\\\\", body) if r.strip()]
    rows = [[text_cmds(c.strip().replace("\\vdots", "\u22ee").replace("\\cdots", "\u22ef"), ctx)
             .replace("\n", " ") for c in r.split("&")] for r in rows]
    if not rows:
        return ""
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    numeric = all(re.fullmatch(r"[-+\d.\u22ee\u22ef\s]*", c) for c in rows[0])
    if numeric:
        # a header-less numeric table (e.g. a matrix): render it as a math array,
        # since GitBook cannot parse a Markdown table with an empty header row
        body = " \\\\\n".join(" & ".join(c.strip().replace("\u22ee", "\\vdots").replace("\u22ef", "\\cdots")
                                        for c in r) for r in rows)
        return "\n\n$$\n\\left[\\begin{array}{" + "c" * n + "}\n" + body + "\n\\end{array}\\right]\n$$\n\n"
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n\n" + "\n".join(lines) + "\n\n"


TOKEN = re.compile(r"\\begin\{(itemize|enumerate|block|alertblock|exampleblock)\}|"
                   r"\\end\{(itemize|enumerate|block|alertblock|exampleblock)\}|\\item(?![a-zA-Z])")


def convert_lists(s, ctx):
    """Turn itemize/enumerate/block into markdown lists and GitBook hints."""
    out, stack = [], []          # stack entries: 'ul', 'ol', 'block'
    pos = 0
    item_open = False

    def list_depth():
        inner = stack[len(stack) - stack[::-1].index("block"):] if "block" in stack else stack
        return sum(1 for x in inner if x in ("ul", "ol"))

    def emit_text(t):
        t = text_cmds(t, ctx)
        depth = list_depth()
        lines = [l.strip() for l in t.split("\n")]
        if depth and item_open:
            indent = "   " * depth
            t = "\n".join((indent + l) if l else "" for l in lines)
        else:
            t = "\n".join(lines)
        out.append(t)

    while True:
        m = TOKEN.search(s, pos)
        if not m:
            emit_text(s[pos:])
            break
        emit_text(s[pos:m.start()])
        pos = m.end()
        if m.group(1):
            env = m.group(1)
            if env in ("itemize", "enumerate"):
                stack.append("ul" if env == "itemize" else "ol")
                item_open = False
                out.append("\n")
            else:
                title = ""
                j = skip_ws(s, pos)
                if j < len(s) and s[j] == "{":
                    title, pos = read_group(s, j)
                stack.append("block")
                item_open = False
                title = text_cmds(title, ctx).strip()
                out.append("\n\n{% hint style=\"info\" %}\n" + (f"**{title}**\n\n" if title else ""))
        elif m.group(2):
            env = m.group(2)
            if stack:
                stack.pop()
            if env in ("itemize", "enumerate"):
                out.append("\n")
                item_open = list_depth() > 0
            else:
                out.append("\n{% endhint %}\n\n")
                item_open = list_depth() > 0
        else:  # \item
            depth = list_depth()
            kind = next((x for x in reversed(stack) if x in ("ul", "ol")), "ul")
            indent = "   " * max(depth - 1, 0)
            out.append("\n" + indent + ("1. " if kind == "ol" else "* "))
            item_open = True
            # optional [label]
            j = skip_ws(s, pos)
            if j < len(s) and s[j] == "[":
                lab, pos = read_group(s, j, "[", "]")
                out.append(f"**{lab}** ")
            else:
                pos = j
    md = "".join(out)
    return md


ITEM = re.compile(r"^( *)(\*|1\.) +(.*)$")


def tidy(md):
    """Normalise whitespace and keep display math / figures inside list items."""
    out, indent = [], None   # indent: content column of the current list item
    in_math = False
    for line in md.split("\n"):
        line = line.rstrip()
        m = ITEM.match(line)
        if in_math:
            out.append((" " * indent if indent else "") + line.strip() if line.strip() else "")
            if line.strip() == "$$":
                in_math = False
            continue
        if m:
            line = f"{m.group(1)}{m.group(2)} {m.group(3).strip()}"
            indent = len(m.group(1)) + len(m.group(2)) + 1
        elif line.strip() == "$$":
            in_math = True
            out.append((" " * indent if indent else "") + "$$")
            continue
        elif line.startswith("{% "):
            indent = None
        elif line and not line.startswith(" ") and not line.startswith("<figure>"):
            indent = None
        elif line.startswith("<figure>") and indent:
            line = " " * indent + line
        out.append(line)
    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md)
    # tight lists: no blank line between consecutive single-line items
    md = re.sub(r"(\n *(?:\*|1\.) [^\n]+)\n\n(?= *(?:\*|1\.) )", r"\1\n", md)
    md = re.sub(r"(\n *(?:\*|1\.) [^\n]+)\n\n(?= *(?:\*|1\.) )", r"\1\n", md)
    # adjacent inline math from the slides ("$\\pm$$0.5$") -> one formula
    md = re.sub(r"\$\$([^$\n]+)\$\$\$\$([^$\n]+)\$\$", r"$$\1 \2$$", md)
    # "$$X =$$" on its own line followed by a display block: fold it into the block
    md = re.sub(r"\n *\$\$([^$\n]*=[^$\n]*)\$\$ *\n\n( *)\$\$\n", lambda m: f"\n{m.group(2)}$$\n{m.group(2)}{m.group(1).strip()}\n", md)
    md = re.sub(r"\{% hint style=\"info\" %\}\s*\{% endhint %\}", "", md)
    md = re.sub(r"\{% hint style=\"info\" %\}\n\n", "{% hint style=\"info\" %}\n", md)
    md = re.sub(r"\n\n\{% endhint %\}", "\n{% endhint %}", md)
    return md.strip() + "\n"
