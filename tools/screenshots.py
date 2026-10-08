"""Capture README screenshots from a running local app (python app.py)."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
OUT = Path(__file__).resolve().parents[1] / "screenshots"
OUT.mkdir(exist_ok=True)
SHOTS = [
    ("arabic-script", "الويفي ما خدامش عندي من البارح", {"width": 1280, "height": 860}, "light"),
    ("arabizi", "Chkoun rbe7 match dyal Raja lbare7?", {"width": 1280, "height": 860}, "light"),
    ("mixed-french", "l'ordinateur dyali ma bghach ycharji, wach mochkil f la batterie?", {"width": 1280, "height": 860}, "dark"),
    ("mobile", "كيفاش نطيب طاجين الدجاج بالزيتون؟", {"width": 390, "height": 1500}, "light"),
]

with sync_playwright() as p:
    browser = p.chromium.launch()
    for name, text, viewport, scheme in SHOTS:
        page = browser.new_page(viewport=viewport, device_scale_factor=2, color_scheme=scheme)
        page.goto(URL)  # the app forces its light theme itself
        page.wait_for_selector("#message textarea")
        page.fill("#message textarea", text)
        page.click("#classify")
        page.wait_for_selector(".result__label", timeout=60000)
        page.wait_for_timeout(900)  # let the bar animation finish
        page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
        print("saved", name, page.inner_text(".result__label"))
        page.close()
    browser.close()
