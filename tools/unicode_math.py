"""Turn simple inline LaTeX into Unicode text, for places that cannot render math.

GitBook shows figure captions as plain text (math in a caption is not rendered),
so caption formulas such as $\\mathcal{A}\\subseteq\\mathcal{D}$ or $T_1,T_2$ are
written with Unicode characters instead: 𝒜⊆𝒟, 𝑇₁,𝑇₂.
"""
import re

GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
         "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
         "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ",
         "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
         "Lambda": "Λ", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω"}
SYMBOLS = {"sim": "∼", "cup": "∪", "cap": "∩", "subseteq": "⊆", "subset": "⊂", "in": "∈",
           "leq": "≤", "geq": "≥", "le": "≤", "ge": "≥", "times": "×", "cdot": "·", "pm": "±",
           "infty": "∞", "to": "→", "neq": "≠", "approx": "≈", ",": " ", ";": " ", "quad": " "}
SCRIPT = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "𝒜ℬ𝒞𝒟ℰℱ𝒢ℋℐ𝒥𝒦ℒℳ𝒩𝒪𝒫𝒬ℛ𝒮𝒯𝒰𝒱𝒲𝒳𝒴𝒵"))
SUB = dict(zip("0123456789+-=()aeoxijkmnpst", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓᵢⱼₖₘₙₚₛₜ"))
SUB.update({"β": "ᵦ", "γ": "ᵧ", "ρ": "ᵨ", "φ": "ᵩ", "χ": "ᵪ"})
SUP = dict(zip("0123456789+-=()niST", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱˢᵀ"))
SUP.update({"*": "*", "'": "′"})


def _bold(ch):
    if "A" <= ch <= "Z":
        return chr(0x1D400 + ord(ch) - ord("A"))
    if "a" <= ch <= "z":
        return chr(0x1D41A + ord(ch) - ord("a"))
    return ch


def _italic(ch):
    if ch == "h":
        return "ℎ"
    if "A" <= ch <= "Z":
        return chr(0x1D434 + ord(ch) - ord("A"))
    if "a" <= ch <= "z":
        return chr(0x1D44E + ord(ch) - ord("a"))
    return ch


def _script(text, table, mark):
    """Sub/superscript text; fall back to mark+text when a character has no Unicode form."""
    plain = text
    if all(c in table for c in plain):
        return "".join(table[c] for c in plain)
    return f"{mark}{plain}" if len(plain) == 1 else f"{mark}({plain})"


def to_unicode(tex):
    s = tex.strip()
    s = re.sub(r"\\(textrm|mathrm|text|operatorname)\{([^{}]*)\}", lambda m: "\x01" + m.group(2) + "\x02", s)
    s = re.sub(r"\\mathbf\{([^{}]*)\}", lambda m: "\x01" + "".join(_bold(c) for c in m.group(1)) + "\x02", s)
    s = re.sub(r"\\mathcal\{([^{}]*)\}", lambda m: "\x01" + "".join(SCRIPT.get(c, c) for c in m.group(1)) + "\x02", s)
    s = re.sub(r"\\bar\{([^{}]*)\}", lambda m: m.group(1) + "\u0304", s)
    s = re.sub(r"\\hat\{([^{}]*)\}", lambda m: m.group(1) + "\u0302", s)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: GREEK.get(m.group(1), SYMBOLS.get(m.group(1), m.group(1))), s)
    s = re.sub(r"\\([,;])", lambda m: SYMBOLS[m.group(1)], s)
    # italicize single Latin letters outside \mathbf/\textrm groups
    out, plain = [], True
    for ch in s:
        if ch == "\x01":
            plain = False
        elif ch == "\x02":
            plain = True
        else:
            out.append(_italic(ch) if plain and ch.isascii() and ch.isalpha() else ch)
    s = "".join(out)
    # sub/superscripts: x_{abc}, x_1, x^{*}, x^S (letters already italicized: map back)
    def plain_text(t):
        return "".join(chr(ord("A") + ord(c) - 0x1D434) if 0x1D434 <= ord(c) <= 0x1D44D else
                       chr(ord("a") + ord(c) - 0x1D44E) if 0x1D44E <= ord(c) <= 0x1D467 else
                       "h" if c == "ℎ" else c for c in t)
    s = re.sub(r"_\{([^{}]*)\}|_(.)", lambda m: _script(plain_text(m.group(1) or m.group(2)), SUB, "_"), s)
    s = re.sub(r"\^\{([^{}]*)\}|\^(.)", lambda m: _script(plain_text(m.group(1) or m.group(2)), SUP, "^"), s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s*([=∼∪∩⊆⊂∈≤≥×→≈])\s*", r" \1 ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"(^|[\s(=])-(?=\d)", "\\1\u2212", s)         # minus sign
    return re.sub(r"\s+", " ", s).strip()


def caption_text(caption):
    """Replace every $$...$$ in a caption by its Unicode form."""
    return re.sub(r"\$\$(.+?)\$\$", lambda m: to_unicode(m.group(1)), caption)
