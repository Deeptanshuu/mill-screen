"""Build banner.html from real model output in predictions.jsonl.

The banner uses the Streamlit demo's own design (streamlit_app.py THEME,
Space Grotesk, the shield-plus icon, the result cards). predictions.jsonl is
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
TOXIC, NON_TOXIC, TEXT, BUTTON, INPUT_BG = "#E53935", "#2E7D32", "#000000", "#000000", "#F8F9FA"

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
                    f'<span class="track"><i style="width:{max(p * 100, 1.2):.1f}%;background:{c}"></i></span>'
                    f'<span class="pct">{p * 100:.1f}%</span></div>')
    badge = "TOXIC" if hit else "NON-TOXIC"
    return f"""<div class="card" style="top:{top}px;border-left-color:{color}">
  <div class="q">&ldquo;{html.escape(r['text'])}&rdquo;</div>
  <div class="hd"><span class="lang">{FLAG[r['lang']]} (detected)</span>
   <span class="badge" style="color:{color};background:{color}1A">{badge}</span></div>
  {''.join(bars[:3])}
</div>"""


es = next(r for r in rows if r["lang"] == "es")
fr = next(r for r in rows if r["lang"] == "fr")

page = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:1280px;height:400px;background:transparent;font-family:'Space Grotesk',sans-serif;color:{TEXT}}}
.b{{position:relative;width:1280px;height:400px;border-radius:16px;overflow:hidden;background:#FFFFFF;border:1px solid #E6E8EB}}
.l{{position:absolute;left:64px;top:70px}}
h1{{display:flex;align-items:center;gap:18px;font-size:92px;font-weight:700;letter-spacing:-2px;line-height:1}}
h1 svg{{flex:none;margin-top:6px}}
p{{margin-top:22px;font-size:22px;color:#333}}
.chips{{margin-top:34px;display:flex;gap:10px;align-items:center}}
.chip{{border:1px solid #D9DCE1;background:{INPUT_BG};border-radius:999px;padding:8px 16px;font-size:15px;font-weight:500}}
.btn{{background:{BUTTON};color:#fff;border-radius:8px;padding:10px 20px;font-size:15px;font-weight:500}}
.r{{position:absolute;left:690px;top:0;width:590px;height:400px;background:{INPUT_BG};border-left:1px solid #EEF0F2}}
.card{{position:absolute;left:44px;width:500px;background:#fff;border-radius:10px;border-left:4px solid;
 box-shadow:0 2px 8px rgba(0,0,0,.10);padding:16px 22px 14px}}
.q{{font-size:15px;color:#222;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.hd{{display:flex;align-items:center;justify-content:space-between;margin:9px 0 10px}}
.lang{{font-size:13px;color:#555}}
.badge{{font-size:13px;font-weight:700;border-radius:6px;padding:3px 9px;letter-spacing:.3px}}
.row{{display:flex;align-items:center;gap:12px;margin-top:6px;font-size:13px}}
.lab{{width:92px;text-align:right;font-weight:600}}
.track{{flex:1;height:12px;background:#F0F1F3;border-radius:3px;overflow:hidden}}
.track i{{display:block;height:100%}}
.pct{{width:48px;font-weight:700}}
</style></head><body><div class="b">
<div class="l">
 <h1>{SHIELD.format(s=78)}Mill</h1>
 <p>A fast, single-pass model for multilingual toxicity</p>
 <div class="chips"><span class="chip">7 languages</span><span class="chip">6 labels</span><span class="chip">XLM-RoBERTa-large</span></div>
</div>
<div class="r">{card(es, 34)}{card(fr, 214)}</div>
</div></body></html>"""

(HERE / "banner.html").write_text(page, encoding="utf-8")
