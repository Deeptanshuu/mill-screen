"""Build architecture.html: the model, stage by stage, in the banner's style.

Shapes and layer lists follow model/language_aware_transformer.py (forward,
lines 187-290). The output bars are the Spanish row of predictions.jsonl, so
they are real model output. The attention grid and bias strip are schematic.
Screenshot the #arch element at deviceScaleFactor 2 to get ../architecture.png.
"""

import json
import math
import pathlib

from build_banner import CUT, LABELS, NON_TOXIC, SHIELD, TOXIC

HERE = pathlib.Path(__file__).parent
rows = [json.loads(line) for line in open(HERE / "predictions.jsonl", encoding="utf-8")]
es = next(r for r in rows if r["lang"] == "es")
N = 8  # tokens drawn


def grey(v):
    c = round(255 - v * 215)
    return f"rgb({c},{c},{c})"


def red(v):
    return f"rgba(229,57,53,{0.12 + v * 0.88:.2f})"


# Schematic values: a smooth content pattern and one per-key language offset.
content = [[0.5 + 0.5 * math.sin(1.7 * i + 0.9 * j) * math.cos(0.6 * i - 1.3 * j) for j in range(N)] for i in range(N)]
bias = [0.15, 0.9, 0.35, 0.7, 0.05, 1.0, 0.5, 0.25]

grid = "".join(f'<i style="background:{grey(v)}"></i>' for row in content for v in row)
strip = "".join(f'<i style="background:{red(v)}"></i>' for v in bias)
tokens = '<i class="cls"></i>' + "<i></i>" * (N - 1)
layers = "".join(f'<i{" class=frozen" if n == 0 else ""}></i>' for n in range(25))


def mini(vals):
    return '<span class="mini">' + "".join(f'<i style="background:{red(v)}"></i>' for r in vals for v in r) + "</span>"


cancel = mini([[0.7] * 4, [0.2] * 4, [1.0] * 4])
survive = mini([[0.7, 0.2, 1.0, 0.4]] * 3)

bars = "".join(
    f'<div class="row"><span class="lab">{name}</span><span class="track">'
    f'<i style="width:{max(es["p"][k] * 100, 0.8):.1f}%;background:{TOXIC if es["p"][k] >= CUT[k] else NON_TOXIC}"></i>'
    f'<b style="left:{CUT[k] * 100:.1f}%"></b></span><span class="pct">{es["p"][k] * 100:.1f}</span></div>'
    for k, name in LABELS)

page = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#fff;font-family:'Space Grotesk',sans-serif;color:#000;padding:20px;width:1640px}}
#arch{{width:1600px;border:1px solid #D8DADD;padding:36px 40px 32px;background:#fff}}
.top{{display:flex;justify-content:space-between;align-items:flex-end;padding-bottom:22px}}
.top h1{{display:flex;align-items:center;gap:10px;font-size:30px;letter-spacing:-.5px}}
.top span{{font-size:13px;color:#6B6F76}}
.flow{{display:flex;align-items:stretch}}
.st{{border-top:2px solid #000;padding-top:12px;flex:none}}
.st.lang{{border-top-color:{TOXIC}}}
.ar{{flex:none;width:28px;display:flex;align-items:center;justify-content:center;color:#6B6F76;font-size:18px}}
.k{{font-size:11px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;color:#6B6F76}}
.k b{{color:#000;margin-right:8px}}
h2{{font-size:19px;font-weight:600;margin:6px 0 12px;line-height:1.2}}
.d{{font-size:13px;line-height:1.5;color:#333}}
code{{font-family:'Space Mono',monospace;font-size:12px}}
.sh{{font-family:'Space Mono',monospace;font-size:12px;color:#6B6F76;white-space:nowrap}}
.li{{display:flex;justify-content:space-between;gap:10px;align-items:baseline;padding:6px 0;border-bottom:1px solid #D8DADD;font-size:13px}}
.li:last-child{{border-bottom:0}}
.li code{{font-weight:700}}
.q{{font-size:14px;font-weight:500;margin-bottom:10px}}
.tok{{display:flex;gap:3px;margin:6px 0 4px}}
.tok i{{width:16px;height:16px;border:1px solid #000}}
.tok i.cls{{background:#000}}
.stack{{display:flex;flex-direction:column-reverse;gap:3px;margin:4px 0 10px}}
.stack i{{height:4px;background:#000}}
.stack i.frozen{{height:10px;background:repeating-linear-gradient(135deg,#9A9DA3 0 2px,#fff 2px 5px);border:1px solid #9A9DA3}}
.core{{display:grid;grid-template-columns:236px 174px 1fr;column-gap:22px}}
.sub{{font-size:11px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:4px}}
.eq{{font-family:'Space Mono',monospace;font-size:12px;font-weight:700;margin-bottom:4px}}
.red{{color:{TOXIC}}}
.grid{{display:grid;grid-template-columns:repeat({N},20px);gap:2px}}
.grid i{{height:20px}}
.axis{{font-size:11px;color:#6B6F76;margin:2px 0 4px}}
.plus{{font-size:20px;font-weight:500;margin:6px 0 2px 80px}}
.steps{{counter-reset:s}}
.steps div{{counter-increment:s;padding:7px 0 7px 26px;border-bottom:1px solid #D8DADD;font-size:13px;position:relative;line-height:1.35}}
.steps div:before{{content:counter(s);position:absolute;left:0;font-family:'Space Mono',monospace;font-size:12px;color:#6B6F76}}
.steps div:last-child{{border-bottom:0}}
.row{{display:flex;align-items:center;gap:10px;height:24px;font-size:13px}}
.lab{{width:88px;color:#333}}
.track{{position:relative;flex:1;height:14px;border-bottom:1px solid #D8DADD}}
.track i{{position:absolute;left:0;bottom:0;height:6px}}
.track b{{position:absolute;top:0;bottom:-1px;width:1px;background:#000}}
.pct{{width:34px;text-align:right;font-weight:600;font-variant-numeric:tabular-nums}}
.foot{{display:grid;grid-template-columns:1.25fr 1fr 1fr;column-gap:44px;margin-top:30px;padding-top:20px;border-top:1px solid #D8DADD}}
.foot h3{{font-size:11px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;color:#6B6F76;margin-bottom:10px}}
.pair{{display:flex;gap:26px}}
.pair div{{font-size:12.5px;line-height:1.45;color:#333;width:50%}}
.mini{{display:grid;grid-template-columns:repeat(4,22px);gap:2px;margin-bottom:8px}}
.mini i{{height:14px}}
.x{{opacity:.35}}
</style></head><body><div id="arch">

<div class="top"><h1>{SHIELD.format(s=28)}Mill, one forward pass</h1>
<span><code>B</code> batch &nbsp;&middot;&nbsp; <code>S</code> tokens &nbsp;&middot;&nbsp; 16 heads of 64 dims &nbsp;&middot;&nbsp; 565M parameters</span></div>

<div class="flow">

<div class="st" style="width:192px">
 <div class="k"><b>01</b>Input</div><h2>A comment and its language</h2>
 <div class="q">Eres un completo idiota y nadie te quiere.</div>
 <div class="d">XLM-R SentencePiece tokens, at most 512. The black square is position 0, <code>[CLS]</code>.</div>
 <div class="tok">{tokens}</div>
 <div class="li"><code>input_ids</code><span class="sh">[B,S]</span></div>
 <div class="li"><code>attention_mask</code><span class="sh">[B,S]</span></div>
 <div class="li"><code class="red">lang_ids</code><span class="sh">[B] &nbsp;es = 3</span></div>
</div>
<div class="ar">&rarr;</div>

<div class="st" style="width:168px">
 <div class="k"><b>02</b>Encoder</div><h2>XLM-RoBERTa-large</h2>
 <div class="stack">{layers}</div>
 <div class="d">24 transformer layers, all trained. The embedding table (hatched, 256M parameters) is frozen.</div>
 <div class="li" style="margin-top:8px"><span>output</span><span class="sh">[B,S,1024]</span></div>
</div>
<div class="ar">&rarr;</div>

<div class="st lang" style="flex:1;min-width:0">
 <div class="k"><b>03</b>Language-conditioned attention</div><h2>One extra attention block, with a per-language bias on the scores</h2>
 <div class="core">
  <div>
   <div class="sub">Content</div>
   <div class="li"><code>q_proj</code><span class="sh">1024&rarr;1024</span></div>
   <div class="li"><code>k_proj</code><span class="sh">1024&rarr;1024</span></div>
   <div class="li"><code>v_proj</code><span class="sh">1024&rarr;1024</span></div>
   <div class="li"><span>split into heads</span><span class="sh">[B,16,S,64]</span></div>
   <div class="sub red" style="margin-top:22px">Language</div>
   <div class="li"><code class="red">lang_embed</code><span class="sh">Embedding(7, 64)</span></div>
   <div class="li"><code class="red">lang_proj</code><span class="sh">64&rarr;1024</span></div>
   <div class="li"><span>LayerNorm, Tanh</span><span class="sh">L [B,16,1,64]</span></div>
  </div>
  <div>
   <div class="eq">q &middot; k<sup>T</sup> / &radic;64</div>
   <div class="axis">keys &rarr; &nbsp; queries &darr;</div>
   <div class="grid">{grid}</div>
   <div class="sh" style="margin-top:5px">[B,16,S,S]</div>
   <div class="plus">+</div>
   <div class="eq red">L &middot; k<sup>T</sup> / &radic;64</div>
   <div class="grid">{strip}</div>
   <div class="sh" style="margin-top:5px">[B,16,1,S]</div>
   <div class="d" style="margin-top:4px;font-size:12px">One value per key, added to every query row.</div>
  </div>
  <div class="steps">
   <div>Mask padded keys to &minus;&infin;</div>
   <div>Softmax along the key axis</div>
   <div>Dropout on the weights</div>
   <div>Weights &middot; <code>v</code>, merge heads</div>
   <div>Linear, LayerNorm, GELU</div>
   <div><span class="sh">[B,S,1024]</span></div>
  </div>
 </div>
</div>
<div class="ar">&rarr;</div>

<div class="st" style="width:176px">
 <div class="k"><b>04</b>Head</div><h2>Classifier on the <code style="font-size:16px">[CLS]</code> row</h2>
 <div class="tok">{tokens}</div>
 <div class="li"><span>row 0</span><span class="sh">[B,1024]</span></div>
 <div class="li"><span>Linear</span><span class="sh">1024&rarr;512</span></div>
 <div class="li"><span>LayerNorm, GELU</span><span class="sh">[B,512]</span></div>
 <div class="li"><span>Linear</span><span class="sh">512&rarr;6</span></div>
 <div class="li"><span>6 sigmoids</span><span class="sh">[B,6]</span></div>
</div>
<div class="ar">&rarr;</div>

<div class="st" style="width:222px">
 <div class="k"><b>05</b>Output</div><h2>Six independent probabilities</h2>
 {bars}
 <div class="d" style="margin-top:10px;font-size:12px">No softmax across labels, so several can fire. A label fires when its probability reaches its own threshold (the tick), tuned on <code>val</code>.</div>
</div>

</div>

<div class="foot">
 <div><h3>Why the bias runs along the keys</h3>
  <div class="pair">
   <div class="x">{cancel}Constant along each row of keys. Softmax is shift-invariant, so it cancels exactly. This was the 2025 bug.</div>
   <div>{survive}Different for every key: <code>L&middot;k_j</code>. It survives softmax and reshapes what every query attends to.</div>
  </div></div>
 <div><h3>Training</h3>
  <div class="li"><span>Loss</span><span class="sh">weighted focal, per class</span></div>
  <div class="li"><span>Frozen</span><span class="sh">embeddings only</span></div>
  <div class="li"><span>Checkpointing</span><span class="sh">use_reentrant=False</span></div>
  <div class="li"><span>Epochs</span><span class="sh">6, best is 5</span></div></div>
 <div><h3>Measured on test</h3>
  <div class="li"><span>AUC, macro</span><span class="sh">0.9852</span></div>
  <div class="li"><span>F1, macro</span><span class="sh">0.8814</span></div>
  <div class="li"><span>Language bias, AUC gain</span><span class="sh">+0.0003, p = 0.588</span></div>
  <div class="li"><span>So <code>lang_ids</code></span><span class="sh">can be left out</span></div></div>
</div>

</div></body></html>"""

(HERE / "architecture.html").write_text(page, encoding="utf-8")
