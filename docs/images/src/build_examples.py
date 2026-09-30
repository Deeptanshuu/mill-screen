"""Build examples.html: three scored comments, drawn from predictions.jsonl.

Same look as the banner (see build_banner.py). predictions.jsonl is real
output of OptimizedToxicityClassifier.predict on the v2 checkpoint; nothing
here is typed in by hand. Screenshot each .fig element at deviceScaleFactor 2
to get ../demo-spanish.png, ../demo-threat.png and ../demo-clean.png.
"""

import html
import json
import pathlib

from build_banner import CUT, FLAG, LABELS, NON_TOXIC, TOXIC

HERE = pathlib.Path(__file__).parent
rows = [json.loads(line) for line in open(HERE / "predictions.jsonl", encoding="utf-8")]


def fig(r, name, width):
    hit = bool(r["fired"])
    bars = []
    for k, label in LABELS:
        p = r["p"][k]
        c = TOXIC if p >= CUT[k] else NON_TOXIC
        bars.append(f'<div class="row"><span class="lab">{label}</span>'
                    f'<span class="track"><i style="width:{max(p * 100, 0.6):.1f}%;background:{c}"></i>'
                    f'<b style="left:{CUT[k] * 100:.1f}%"></b></span>'
                    f'<span class="pct">{p * 100:.1f}</span></div>')
    return f"""<div class="fig" id="{name}" style="width:{width}px">
  <div class="hd"><span>{FLAG[r['lang']]}</span>
   <span style="color:{TOXIC if hit else NON_TOXIC}">{'toxic' if hit else 'non-toxic'}</span></div>
  <div class="q">{html.escape(r['text'])}</div>
  {''.join(bars)}
  <div class="note">probability, % &nbsp;&middot;&nbsp; tick marks the label's threshold</div>
</div>"""


def pick(text):
    return next(r for r in rows if r["text"].startswith(text))


page = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#fff;font-family:'Space Grotesk',sans-serif;color:#000;padding:20px}}
.fig{{background:#fff;border:1px solid #D8DADD;padding:28px 32px 20px;margin-bottom:20px}}
.hd{{display:flex;justify-content:space-between;font-size:12px;font-weight:600;letter-spacing:1.2px;
 text-transform:uppercase;color:#6B6F76}}
.q{{margin:8px 0 18px;font-size:21px;font-weight:500}}
.row{{display:flex;align-items:center;gap:14px;height:26px;font-size:14px}}
.lab{{width:104px;color:#333}}
.track{{position:relative;flex:1;height:14px;border-bottom:1px solid #D8DADD}}
.track i{{position:absolute;left:0;bottom:0;height:6px}}
.track b{{position:absolute;top:0;bottom:-1px;width:1px;background:#000}}
.pct{{width:40px;text-align:right;font-weight:600;font-variant-numeric:tabular-nums}}
.note{{margin-top:14px;font-size:11px;color:#6B6F76;text-align:right}}
</style></head><body>
{fig(pick("Eres un completo"), "demo-spanish", 800)}
{fig(pick("I know where you live"), "demo-threat", 560)}
{fig(pick("Thanks for the fix"), "demo-clean", 560)}
</body></html>"""

(HERE / "examples.html").write_text(page, encoding="utf-8")
