"""Build architecture.html: a block diagram of the model, bottom to top.

Layer names and shapes follow model/language_aware_transformer.py (__init__ at
62-112, forward at 187-290). The output bars are the Spanish row of
predictions.jsonl, so they are real model output; the small score grids are
schematic. Screenshot the #arch element at deviceScaleFactor 2 to get
../architecture.png.
"""

import json
import math
import pathlib

from build_banner import CUT, LABELS, NON_TOXIC, TOXIC

HERE = pathlib.Path(__file__).parent
rows = [json.loads(line) for line in open(HERE / "predictions.jsonl", encoding="utf-8")]
es = next(r for r in rows if r["lang"] == "es")

INK, MUTED, LINE = "#000000", "#6B6F76", "#B4B7BD"
PLAIN = ("#FAFAF8", "#B4B7BD")
ATTN = ("#FFF4CC", "#D2AE3A")
LANG = ("#FDE7E6", TOXIC)
HEAD = ("#E3F0E4", NON_TOXIC)
FROZEN = ("url(#hatch)", "#8E9299")
out = []


def rect(x, y, w, h, fill, stroke, dash=False, r=5, sw=1.4):
    d = ' stroke-dasharray="2 4" stroke-linecap="round"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=15, weight=500, fill=INK, anchor="middle", cls="", style=""):
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
               f'text-anchor="{anchor}" class="{cls}" style="{style}">{s}</text>')


def box(cx, y, w, h, title, sub=None, kind=PLAIN, stack=0, mono=False):
    """A block centred on cx. stack > 0 draws offset copies behind it for depth."""
    x = cx - w / 2
    for n in range(stack, 0, -1):
        rect(x + 7 * n, y - 7 * n, w, h, "#FFFFFF" if kind is FROZEN else kind[0], kind[1])
    if kind is FROZEN:
        rect(x, y, w, h, "#FFFFFF", kind[1])
    rect(x, y, w, h, kind[0], kind[1])
    if sub:
        text(cx, y + h / 2 - 3, title, cls="m" if mono else "")
        text(cx, y + h / 2 + 15, sub, size=11.5, weight=400, fill=MUTED, cls="m")
    else:
        text(cx, y + h / 2 + 5, title, cls="m" if mono else "")


def line(pts, color="#333", arrow=True, sw=1.6):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    m = f' marker-end="url(#a{"r" if color == TOXIC else "k"})"' if arrow else ""
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{m}/>')


def tag(x, y, s, anchor="start", fill=MUTED, size=12):
    text(x, y, s, size=size, weight=400, fill=fill, anchor=anchor, cls="m")


def grid(x, y, n, cell, vals, color):
    for i in range(n):
        for j in range(len(vals[i])):
            v = vals[i][j]
            f = f"rgba(229,57,53,{0.15 + 0.85 * v:.2f})" if color == "red" else f"rgb({round(250 - v * 200)},{round(245 - v * 205)},{round(225 - v * 215)})"
            out.append(f'<rect x="{x + j * cell}" y="{y + i * cell}" width="{cell - 1.5}" height="{cell - 1.5}" fill="{f}"/>')


content = [[0.5 + 0.5 * math.sin(1.7 * i + 0.9 * j) * math.cos(0.6 * i - 1.3 * j) for j in range(5)] for i in range(5)]

# ---------------------------------------------------------------- header
text(40, 62, "Mill, one forward pass", size=30, weight=700, anchor="start", style="letter-spacing:-.5px")
tag(40, 90, "B batch  ·  S tokens  ·  16 heads of 64 dims  ·  565M parameters", size=13)
for i, (kind, name) in enumerate([(PLAIN, "layer, trained"), (FROZEN, "frozen"), (ATTN, "attention scores"),
                                  (LANG, "language path"), (HEAD, "classifier")]):
    rect(40, 122 + i * 26, 22, 15, kind[0], kind[1], r=3, sw=1.2)
    text(72, 134 + i * 26, name, size=13, weight=400, anchor="start", fill="#333")

# ---------------------------------------------------------------- encoder panel
LX = 370
rect(170, 350, 400, 440, "#F4F4F1", "#D8DADD", r=10, sw=1.2)
text(552, 378, "XLM-RoBERTa-large", size=14, weight=700, anchor="end", fill="#333")
for n in (2, 1):
    rect(232 + 6 * n, 404 - 6 * n, 276, 272, "none", "#9A9DA3", dash=True, r=8)
rect(232, 404, 276, 272, "#F4F4F1", "#6B6F76", dash=True, r=8)
text(540, 548, "×24", size=17, weight=600, anchor="middle")
box(LX, 614, 220, 44, "Multi-head self-attention", "16 heads")
box(LX, 558, 220, 34, "Add &amp; LayerNorm")
box(LX, 486, 220, 48, "Feed-forward", "1024 → 4096 → 1024")
box(LX, 428, 220, 34, "Add &amp; LayerNorm")
box(LX, 716, 220, 50, "Embedding", "frozen · 256M parameters", kind=FROZEN, stack=2)
box(LX, 276, 220, 44, "Enc. hidden states", stack=3)
box(LX, 830, 220, 40, "input_ids", mono=True)
box(95, 830, 150, 40, "attention_mask", mono=True)
box(LX, 900, 300, 40, "XLM-R SentencePiece tokenizer")
text(LX, 972, "“Eres un completo idiota y nadie te quiere.”", size=14, weight=500, fill="#333")
line([(LX, 955), (LX, 942)])
line([(LX, 900), (LX, 872)])
line([(220, 920), (95, 920), (95, 872)])
line([(LX, 830), (LX, 768)])
tag(LX + 10, 806, "[B,S]")
line([(LX, 709), (LX, 678)])
line([(LX, 614), (LX, 594)])
line([(LX, 558), (LX, 536)])
line([(LX, 486), (LX, 464)])
line([(LX, 428), (LX, 322)])
line([(95, 830), (95, 636), (258, 636)])
tag(105, 628, "[B,S]")

# ---------------------------------------------------------------- head panel
RX = 1010
rect(790, 112, 640, 678, "#F4F4F1", "#D8DADD", r=10, sw=1.2)
text(1412, 140, "Language-aware head", size=14, weight=700, anchor="end", fill="#333")

# hidden states travel from the encoder into the head
line([(501, 291), (700, 291), (700, 812), (RX, 812), (RX, 768)])
tag(710, 520, "[B,S,1024]")

box(RX, 716, 250, 50, "q, k, v projections", "3 × Linear 1024 → 1024", stack=2)
line([(RX, 709), (RX, 690)])

# content scores, drawn as a stack of per-head matrices
rect(RX - 125, 618, 250, 70, *ATTN)
for n in (2, 1, 0):
    rect(RX - 110 + 6 * n - 2, 628 - 5 * n - 2 + 6, 49, 49, "#FFFFFF", ATTN[1], r=2, sw=1)
    grid(RX - 110 + 6 * n, 634 - 5 * n, 5, 9, content, "grey")
text(RX + 34, 648, "content scores")
text(RX + 34, 667, "q·kᵀ / √64   [B,16,S,S]", size=11.5, weight=400, fill=MUTED, cls="m")
line([(RX, 618), (RX, 592)])

# the sum node
out.append(f'<circle cx="{RX}" cy="577" r="14" fill="#fff" stroke="{TOXIC}" stroke-width="1.6"/>')
text(RX, 584, "+", size=20, weight=600, fill=TOXIC)
line([(RX, 563), (RX, 538)])

box(RX, 494, 250, 42, "Mask padded keys to −∞")
line([(RX, 494), (RX, 476)])
box(RX, 430, 250, 44, "Softmax over keys", "then dropout", kind=ATTN)
line([(RX, 430), (RX, 412)])
box(RX, 366, 250, 44, "Weights · v, merge heads")
line([(RX, 366), (RX, 348)])
box(RX, 296, 250, 50, "post_attention", "Linear · LayerNorm · GELU", mono=True)
line([(RX, 296), (RX, 278)])
tag(RX + 10, 290, "[B,S,1024]")

# keep row 0
rect(RX - 125, 232, 250, 44, *PLAIN)
for j in range(7):
    out.append(f'<rect x="{RX - 110 + j * 13}" y="246" width="10" height="16" fill="{INK if j == 0 else "#fff"}" stroke="{INK}" stroke-width="1"/>')
text(RX + 52, 259, "keep the [CLS] row")
line([(RX, 232), (RX, 214)])
tag(RX + 10, 226, "[B,1024]")
box(RX, 162, 250, 50, "classifier", "1024 → 512 → 6", kind=HEAD, mono=True)
line([(RX, 162), (RX, 92)])
tag(RX + 10, 132, "6 logits  [B,6]")

# v skips the scores and rejoins at the weighted sum
line([(RX - 125, 741), (RX - 165, 741), (RX - 165, 388), (RX - 127, 388)])
tag(RX - 173, 560, "v", anchor="end")

# ---------------------------------------------------------------- language branch
GX = 1300
rect(1178, 540, 236, 240, "none", TOXIC, dash=True, r=8)
text(1404, 532, "language conditioning", size=12, weight=600, anchor="end", fill=TOXIC)
box(GX, 716, 200, 50, "lang_embed", "Embedding(7, 64)", kind=LANG, mono=True)
box(GX, 640, 200, 50, "lang_proj", "Linear 64 → 1024, Tanh", kind=LANG, mono=True)
rect(1192, 554, 210, 46, *LANG)
grid(1204, 571, 1, 12, [[0.15, 0.9, 0.35, 0.7, 0.05]], "red")
text(1336, 573, "language bias")
text(1336, 591, "L·kᵀ / √64", size=11.5, weight=400, fill=MUTED, cls="m")
box(GX, 830, 200, 40, "lang_ids", mono=True, kind=LANG)
tag(GX, 890, "en 0 · ru 1 · tr 2 · es 3 · fr 4 · it 5 · pt 6", anchor="middle", size=11.5)
line([(GX, 830), (GX, 768)], color=TOXIC)
tag(GX + 10, 806, "[B]")
line([(GX, 716), (GX, 692)], color=TOXIC)
line([(GX, 640), (GX, 602)], color=TOXIC)
tag(GX + 10, 626, "L [B,16,1,64]")
line([(1192, 577), (RX + 16, 577)], color=TOXIC)
tag(1108, 568, "[B,16,1,S]", anchor="middle")
# k also feeds the bias
line([(RX + 125, 741), (1162, 741), (1162, 590), (1190, 590)])
tag(1154, 668, "k", anchor="end")

# ---------------------------------------------------------------- output
box(RX, 46, 250, 44, "6 independent sigmoids", kind=HEAD, stack=3)
line([(RX + 146, 54), (1206, 54)])
bx, by = 1216, 14
for i, (k, name) in enumerate(LABELS):
    p, y = es["p"][k], by + i * 15
    text(bx, y + 10, name, size=11.5, weight=400, anchor="start", fill="#333")
    out.append(f'<line x1="{bx + 92}" y1="{y + 11}" x2="{bx + 232}" y2="{y + 11}" stroke="#D8DADD"/>')
    out.append(f'<rect x="{bx + 92}" y="{y + 6}" width="{max(p * 140, 1.5):.1f}" height="5" fill="{TOXIC if p >= CUT[k] else NON_TOXIC}"/>')
    out.append(f'<line x1="{bx + 92 + CUT[k] * 140:.1f}" y1="{y + 1}" x2="{bx + 92 + CUT[k] * 140:.1f}" y2="{y + 12}" stroke="#000"/>')
    text(bx + 272, y + 10, f"{p * 100:.1f}", size=11.5, weight=600, anchor="end")

svg = "\n".join(out)
page = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono&display=swap" rel="stylesheet">
<style>body{{margin:20px;background:#fff}}svg{{font-family:'Space Grotesk',sans-serif;display:block;border:1px solid #D8DADD;background:#fff}}
.m{{font-family:'Space Mono',monospace}}text.m[font-size="15"]{{font-size:13.5px;font-weight:700}}</style></head><body>
<svg id="arch" xmlns="http://www.w3.org/2000/svg" width="1520" height="1024" viewBox="0 -24 1520 1024">
<defs>
 <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
  <rect width="6" height="6" fill="#FAFAF8"/><line x1="0" y1="0" x2="0" y2="6" stroke="#E3E5E8" stroke-width="2"/></pattern>
 <marker id="ak" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker>
 <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{TOXIC}"/></marker>
</defs>
{svg}
</svg></body></html>"""

(HERE / "architecture.html").write_text(page, encoding="utf-8")
