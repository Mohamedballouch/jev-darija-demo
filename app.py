"""Jev-Darija demo: classify a Moroccan Darija message into one of 25 topics.

The model and its question set are downloaded from the Hugging Face Hub
(mballouch/jev-darija) on first start and cached locally afterwards.
"""
import html
import json
import os
import re
import time
from urllib.parse import quote

import gradio as gr
import laya
import torch
from huggingface_hub import hf_hub_download

MODEL_ID = os.environ.get("JEV_DARIJA_MODEL", "mballouch/jev-darija")
DEVICE = os.environ.get("JEV_DARIJA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN, HEAD_MAX_LEN = 2048, 768  # fits the full 25-option question without truncation
TOP_K = 5

EXAMPLE_GROUPS = {
    "Arabic script": [
        "الويفي ما خدامش عندي من البارح",
        "كيفاش نطيب طاجين الدجاج بالزيتون؟",
        "بغيت نكري شقة فيها جوج بيوت فكازا",
        "شحال كاتاخد البنكة على تحويل الفلوس لبرا؟",
    ],
    "Arabizi": [
        "Chkoun rbe7 match dyal Raja lbare7?",
        "bghit n7jez billet dyal train l Marrakech had sbt",
    ],
    "Mixed with French": [
        "l'ordinateur dyali ma bghach ycharji, wach mochkil f la batterie?",
        "3andi rendez-vous 3nd tbib l9alb ghedda, khassni nsowlo 3la dwa",
    ],
}

ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")
FRENCH_HINT = re.compile(r"\b(le|la|les|des|du|de|un|une|est|pour|avec|l'|d')\b|[éèêàùç]", re.IGNORECASE)

print(f"Loading {MODEL_ID} on {DEVICE}…", flush=True)
agent = laya.load(MODEL_ID, device=DEVICE)
with open(hf_hub_download(MODEL_ID, "questions.json"), encoding="utf-8") as handle:
    QUESTIONS = json.load(handle)
DOMAINS = list(QUESTIONS["intent"]["criteria"])
# One warm-up call so the first real message is as fast as the rest.
agent.system_one(EXAMPLE_GROUPS["Arabic script"][0], QUESTIONS, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)


def pretty(label):
    return label.replace("_and_", " & ").replace("_", " ")


def writing_style(text):
    has_arabic, has_latin = bool(ARABIC.search(text)), bool(LATIN.search(text))
    if has_arabic and has_latin:
        return "Arabic script mixed with Latin"
    if has_arabic:
        return "Arabic script"
    if has_latin:
        return "Arabizi with French" if FRENCH_HINT.search(text) else "Arabizi"
    return "No letters found"


def percent(p):
    return "<1%" if p < 0.005 else f"{p:.0%}"


def empty_state():
    return (
        '<div class="result result--empty">'
        "<p>Type or paste a message, then press <strong>Classify</strong>.</p>"
        f"<p>The model chooses among {len(DOMAINS)} topics, from finance and telecoms to health and sports.</p>"
        "</div>"
    )


def classify(text):
    text = (text or "").strip()
    if not text:
        return empty_state()
    started = time.perf_counter()
    answer = agent.system_one(text, QUESTIONS, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)["answers"]["intent"]
    elapsed_ms = (time.perf_counter() - started) * 1000
    ranked = sorted(answer["probabilities"].items(), key=lambda item: item[1], reverse=True)[:TOP_K]
    top_label, top_p = ranked[0]
    rows = "".join(
        f'<li class="bar{" bar--top" if i == 0 else ""}">'
        f'<span class="bar__name">{html.escape(pretty(label))}</span>'
        f'<span class="bar__value">{percent(p)}</span>'
        f'<span class="bar__track"><span class="bar__fill" style="--p:{max(p, 0.004):.4f}"></span></span>'
        "</li>"
        for i, (label, p) in enumerate(ranked)
    )
    return (
        '<div class="result">'
        '<p class="result__kicker">This message is about</p>'
        f'<p class="result__label">{html.escape(pretty(top_label))}</p>'
        f'<p class="result__confidence">{top_p:.0%} probability</p>'
        f'<ol class="bars" aria-label="Top {TOP_K} topics">{rows}</ol>'
        '<dl class="meta">'
        f"<div><dt>Writing</dt><dd>{writing_style(text)}</dd></div>"
        f"<div><dt>Time</dt><dd>{elapsed_ms:.0f} ms on {'GPU' if DEVICE == 'cuda' else 'CPU'}</dd></div>"
        "</dl>"
        "</div>"
    )


def zellige(line, joint, line_op, fill_op, joint_op):
    """A 96 px zellige tile: eight-point stars linked by bars and diamond joints, as a CSS url()."""
    T, C, H = 96, 48, 17            # tile size, centre, half-size of each square in a star
    tip = round(H * 2 ** 0.5)       # distance from a star's centre to its axis tips

    def star(cx, cy):
        square = f'x="{cx-H}" y="{cy-H}" width="{2*H}" height="{2*H}"'
        return f'<rect {square}/><rect {square} transform="rotate(45 {cx} {cy})"/>'

    stars = "".join(star(x, y) for x, y in [(C, C), (0, 0), (T, 0), (0, T), (T, T)])
    bars = "".join(f'<line x1="{a}" y1="{b}" x2="{c}" y2="{d}"/>' for a, b, c, d in [
        (C + tip, C, T - 5, C), (C - tip, C, 5, C), (C, C + tip, C, T - 5), (C, C - tip, C, 5),
        (tip, 0, C - 5, 0), (T - tip, 0, C + 5, 0), (tip, T, C - 5, T), (T - tip, T, C + 5, T),
        (0, tip, 0, C - 5), (0, T - tip, 0, C + 5), (T, tip, T, C - 5), (T, T - tip, T, C + 5)])
    joints = "".join(f'<rect x="{x-5}" y="{y-5}" width="10" height="10" transform="rotate(45 {x} {y})"/>'
                     for x, y in [(C, 0), (0, C), (T, C), (C, T)])
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{T}" height="{T}" viewBox="0 0 {T} {T}">'
           f'<g fill="{line}" fill-opacity="{fill_op}" stroke="{line}" stroke-opacity="{line_op}" stroke-width="1.1">{stars}{bars}</g>'
           f'<g fill="{joint}" fill-opacity="{joint_op}" stroke="{line}" stroke-opacity="{line_op}" stroke-width="1.1">{joints}</g></svg>')
    return 'url("data:image/svg+xml,' + quote(svg) + '")'


HEAD = """
<script>
  // Always use the light theme. Gradio adds this head content after it has applied the
  // system dark-mode setting, so remove its "dark" class now and whenever it is re-added.
  (function () {
    const keepLight = () => document.body.classList.remove("dark");
    keepLight();
    new MutationObserver(keepLight).observe(document.body, { attributes: true, attributeFilter: ["class"] });
    window.__jevDarijaLight = true;
  })();
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Readex+Pro:wght@300;400;500;600;700&display=swap" rel="stylesheet">
"""

CSS = """
:root, .light, .dark {
  --ink: #1A1D3A; --muted: #5F6383; --page: #F3F4FA; --panel: #FFFFFF;
  --line: #E3E5F1; --accent: #4A3FCF; --accent-ink: #FFFFFF; --saffron: #E8A317;
  --zellige: __ZELLIGE_LIGHT__;
}
.gradio-container, .gradio-container * { font-family: "Readex Pro", system-ui, sans-serif !important; }
body {
  color: var(--ink) !important;
  background-color: var(--page) !important;
  background-image: linear-gradient(to bottom, color-mix(in srgb, var(--page) 15%, transparent) 0, color-mix(in srgb, var(--page) 70%, transparent) 300px, var(--page) 620px), var(--zellige) !important;
  background-size: auto, 96px 96px !important; background-attachment: scroll !important;
}
gradio-app, .main, .wrap, .gradio-container, .gradio-container > .main { background: transparent !important; }
.gradio-container { color: var(--ink); max-width: 1080px !important; margin: 0 auto !important; }
footer { display: none !important; }

#masthead { padding: 36px 4px 18px; }
#masthead .wordmark { font-size: 1.6rem; font-weight: 600; letter-spacing: -0.01em; color: var(--accent); margin: 0 0 12px; }
#masthead h1 { margin: 0; color: var(--ink); line-height: 1.25; }
#masthead .h-ar { display: table; font-size: 2.7rem; font-weight: 600; }
#masthead .h-lat { display: block; font-size: 2.35rem; font-weight: 300; letter-spacing: -0.01em; margin-top: 2px; }
#masthead .sub { font-size: 1.05rem; color: var(--muted); margin: 16px 0 0; max-width: 62ch; line-height: 1.55; }
#masthead .sub { background: color-mix(in srgb, var(--page) 80%, transparent); display: inline-block; }

#message textarea {
  font-size: 1.25rem !important; line-height: 1.6 !important; color: var(--ink) !important;
  unicode-bidi: plaintext; text-align: start; background: var(--page) !important;
  border: 1px solid var(--line) !important; border-radius: 10px !important; padding: 14px 16px !important;
}
#message textarea:focus { outline: 2px solid var(--accent) !important; outline-offset: 1px; }
#classify { background: var(--accent) !important; color: var(--accent-ink) !important; border: none !important; font-weight: 500 !important; font-size: 1rem !important; }
#clear { background: transparent !important; color: var(--muted) !important; border: 1px solid var(--line) !important; }
#classify:focus-visible, #clear:focus-visible { outline: 2px solid var(--saffron) !important; outline-offset: 2px; }
[id^="examples-"] .gallery-item, [id^="examples-"] button { unicode-bidi: plaintext; font-size: 0.98rem !important; }
[id^="examples-"] { margin-top: 4px !important; }

.result { color: var(--ink); background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 24px 26px; }
.result--empty p { color: var(--muted); font-size: 1.02rem; line-height: 1.6; margin: 0 0 10px; }
.result__kicker { color: var(--muted); font-size: 0.95rem; margin: 0; }
.result__label {
  font-size: 2.3rem; font-weight: 600; line-height: 1.15; margin: 4px 0 6px; letter-spacing: -0.015em;
  display: inline-block; padding-bottom: 6px; border-bottom: 4px solid var(--saffron);
}
.result__confidence { color: var(--muted); margin: 6px 0 22px; font-size: 1rem; }
.bars { list-style: none; padding: 0; margin: 0; display: grid; gap: 11px; }
.bar { display: grid; grid-template-columns: 1fr auto; gap: 6px 12px; align-items: baseline; }
.bar__name { font-size: 1rem; }
.bar__value { font-size: 0.95rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.bar__track { grid-column: 1 / -1; height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; }
.bar__fill { display: block; height: 100%; width: calc(var(--p) * 100%); background: var(--muted); opacity: 0.55; border-radius: 4px; animation: fill 600ms cubic-bezier(.2,.7,.2,1) both; transform-origin: left; }
.bar--top .bar__name { font-weight: 500; }
.bar--top .bar__fill { background: var(--accent); opacity: 1; }
@keyframes fill { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@media (prefers-reduced-motion: reduce) { .bar__fill { animation: none; } }
.meta { display: flex; flex-wrap: wrap; gap: 8px 28px; margin: 24px 0 0; padding-top: 16px; border-top: 1px solid var(--line); }
.meta dt { color: var(--muted); font-size: 0.85rem; }
.meta dd { margin: 2px 0 0; font-size: 0.98rem; }

#note { color: var(--muted); font-size: 0.92rem; line-height: 1.6; padding: 4px; }
#note a { color: var(--accent); margin: 0 !important; padding: 0 !important; }
@media (min-width: 900px) { #message textarea { min-height: 430px !important; } }
@media (max-width: 720px) { #masthead .h-ar { font-size: 1.9rem; } #masthead .h-lat { font-size: 1.6rem; } .result__label { font-size: 1.8rem; } }
"""

CSS = (CSS.replace("__ZELLIGE_LIGHT__", zellige("#4A3FCF", "#E8A317", 0.26, 0.05, 0.35)))

with gr.Blocks(title="Jev-Darija | Darija topic classifier") as demo:
    gr.HTML(
        '<header id="masthead">'
        '<p class="wordmark">Jev-Darija</p>'
        '<h1><span class="h-ar" lang="ary" dir="rtl">شنو الموضوع ديال هاد الميساج؟</span>'
        '<span class="h-lat" lang="ary-Latn">chno lmawdou3 dyal had lmessage?</span></h1>'
        '<p class="sub">Paste a Moroccan Darija message in Arabic script, Arabizi, or mixed with French. '
        f"The model tells you which of its {len(DOMAINS)} topics it is about.</p>"
        "</header>"
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=6):
            message = gr.Textbox(
                label="Message", lines=5, max_lines=10, elem_id="message",
                placeholder="مثلا: الويفي ما خدامش عندي من البارح  /  e.g. Chkoun rbe7 match dyal Raja lbare7?",
            )
            with gr.Row():
                classify_button = gr.Button("Classify", variant="primary", elem_id="classify", scale=3)
                clear_button = gr.Button("Clear", elem_id="clear", scale=1)
        with gr.Column(scale=5):
            result = gr.HTML(empty_state())
    # Below both columns, so on a phone the result sits right under the Classify button.
    with gr.Row(equal_height=False):
        for index, (style, texts) in enumerate(EXAMPLE_GROUPS.items()):
            with gr.Column(scale=2 if index == 0 else 1, min_width=280):
                gr.Examples(texts, inputs=message, label=style, elem_id=f"examples-{index}")
    gr.HTML(
        '<p id="note">Model: <a href="https://huggingface.co/mballouch/jev-darija">mballouch/jev-darija</a> '
        "(Laya multilingual, 322M parameters, fine-tuned on Darija). "
        'Data: <a href="https://huggingface.co/datasets/mballouch/jev-darija-synthetic">mballouch/jev-darija-synthetic</a>. '
        "This is an experimental fan project, not a research or commercial project.</p>"
    )

    classify_button.click(classify, inputs=message, outputs=result)
    message.submit(classify, inputs=message, outputs=result)
    clear_button.click(lambda: ("", empty_state()), outputs=[message, result])

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate", radius_size="md"),
        css=CSS, head=HEAD,
        server_name=os.environ.get("HOST", "127.0.0.1"), server_port=int(os.environ.get("PORT", "7860")),
    )
