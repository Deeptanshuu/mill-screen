"""Build banner.html from real model output in predictions.jsonl.

The banner uses the Streamlit demo's own design (streamlit_app.py THEME,
Space Grotesk, the shield-plus icon). predictions.jsonl is
the output of OptimizedToxicityClassifier.predict on the v2 checkpoint with the
per-class thresholds; no number on the cards is typed in by hand.

Render banner.html in a headless browser at 1280x400 with deviceScaleFactor 2
to get ../banner.png.
"""

import html
import json
import pathlib

HERE = pathlib.Path(__file__).parent
rows = [json.loads(line) for line in open(HERE / "predictions.jsonl", encoding="utf-8")]

# streamlit_app.py THEME
TOXIC, NON_TOXIC, TEXT = "#E53935", "#2E7D32", "#000000"

LABELS = [("toxic", "Toxic"), ("severe_toxic", "Severe Toxic"), ("obscene", "Obscene"),
          ("threat", "Threat"), ("insult", "Insult"), ("identity_hate", "Identity Hate")]
# Same order and values as OptimizedToxicityClassifier.TUNED_THRESHOLDS.
CUT = dict(zip([k for k, _ in LABELS], [0.4724, 0.4724, 0.5276, 0.5276, 0.5643, 0.5643], strict=True))
FLAG = {"es": "Spanish", "fr": "French", "en": "English", "ru": "Russian"}

SHIELD = ('<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
          'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
          '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1'
          'c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'
          '<path d="M9 12h6"/><path d="M12 9v6"/></svg>')


def card(r, top):
    hit = bool(r["fired"])
    color = TOXIC if hit else NON_TOXIC
    bars = []
    for k, name in sorted(LABELS, key=lambda kv: -r["p"][kv[0]]):
        p = r["p"][k]
        c = TOXIC if p >= CUT[k] else NON_TOXIC
        bars.append(f'<div class="row"><span class="lab">{name}</span>'
                    f'<span class="track"><i style="width:{max(p * 100, 0.6):.1f}%;background:{c}"></i>'
                    f'<b style="left:{CUT[k] * 100:.1f}%"></b></span>'
                    f'<span class="pct">{p * 100:.1f}</span></div>')
    verdict = "toxic" if hit else "non-toxic"
    return f"""<div class="ex" style="top:{top}px">
  <div class="hd"><span>{FLAG[r['lang']]}</span><span style="color:{color}">{verdict}</span></div>
  <div class="q">{html.escape(r['text'])}</div>
  {''.join(bars[:3])}
</div>"""


es = next(r for r in rows if r["lang"] == "es")
fr = next(r for r in rows if r["lang"] == "fr")

page = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:1280px;height:400px;background:transparent;font-family:'Space Grotesk',sans-serif;color:{TEXT}}}
.b{{position:relative;width:1280px;height:400px;overflow:hidden;background:#FFFFFF;border:1px solid #D8DADD}}
.l{{position:absolute;left:64px;top:92px}}
h1{{display:flex;align-items:center;gap:18px;font-size:92px;font-weight:700;letter-spacing:-2px;line-height:1}}
h1 svg{{flex:none;margin-top:6px}}
p{{margin-top:22px;font-size:22px;color:#222}}
.meta{{margin-top:14px;font-size:15px;color:#6B6F76}}
.r{{position:absolute;left:700px;top:0;width:580px;height:400px;border-left:1px solid #D8DADD}}
.ex{{position:absolute;left:48px;width:484px}}
.hd{{display:flex;justify-content:space-between;font-size:12px;font-weight:600;letter-spacing:1.2px;
 text-transform:uppercase;color:#6B6F76}}
.q{{margin:7px 0 14px;font-size:19px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.row{{display:flex;align-items:center;gap:14px;height:22px;font-size:13px}}
.lab{{width:96px;color:#333}}
.track{{position:relative;flex:1;height:14px;border-bottom:1px solid #D8DADD}}
.track i{{position:absolute;left:0;bottom:0;height:6px}}
.track b{{position:absolute;top:0;bottom:-1px;width:1px;background:#000}}
.pct{{width:36px;text-align:right;font-weight:600;font-variant-numeric:tabular-nums}}
.rule{{position:absolute;left:48px;width:484px;top:196px;border-top:1px solid #D8DADD}}
.note{{position:absolute;right:48px;bottom:16px;font-size:11px;color:#6B6F76}}
</style></head><body><div class="b">
<div class="l">
 <h1>{SHIELD.format(s=78)}Mill</h1>
 <p>A fast, single-pass model for multilingual toxicity</p>
 <div class="meta">7 languages &nbsp;/&nbsp; 6 labels &nbsp;/&nbsp; XLM-RoBERTa-large</div>
</div>
<div class="r">{card(es, 44)}<div class="rule"></div>{card(fr, 218)}
 <div class="note">probability, % &nbsp;&middot;&nbsp; tick marks the label's threshold</div></div>
</div></body></html>"""

(HERE / "banner.html").write_text(page, encoding="utf-8")
