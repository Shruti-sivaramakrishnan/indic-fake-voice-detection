"""
Theming and self-contained HTML/CSS/JS snippets for the Streamlit demo (app.py).

Palette: yellow + white + black, with two semantic accents (success=green,
warning/error=amber) — every text/background pair below was checked against
WCAG contrast requirements (4.5:1 for text, 3:1 for UI-component borders)
before being used; see the comment above PALETTE.

Split between two delivery mechanisms, on purpose:

  * Static page chrome (base background, pill-button hover sweep on
    st.button(type="primary"), st.pills reskin, alert cards) is injected as
    <style> into the MAIN document via st.markdown(unsafe_allow_html=True).
    This is the only way to theme Streamlit's own real widgets — an iframe
    (components.v1.html) can't reach outside itself to style them, and CSS
    :hover/transitions on real DOM elements work identically regardless of
    where the <style> tag lives.

  * Everything animated/decorative that does NOT need to feed a value back to
    Python — the landing illustration + panic dramatization, the hamster-wheel
    loader, the WhatsApp "coming soon" scroll-reveal card — is built as a
    fully self-contained HTML document and rendered via
    st.components.v1.html(), per the brief's instruction, since these need
    real CSS keyframe animation (and one small JS scroll listener) that has
    no equivalent in native Streamlit widgets.

  * The one deliberate exception: the decorative background pattern is STATIC
    (no hover/keyframes), and a page-spanning background is a main-document
    concern by nature — an iframe is a bounded rectangle, not a layer that can
    sit behind the rest of the page. So it lives in the main-document CSS
    (BASE_STYLES) via a low-alpha repeating-radial-gradient, not in a
    components.html snippet. No actual pattern asset files were shared in
    this conversation, so this is a generated substitute — swap
    `_BACKGROUND_PATTERN_CSS` for a real asset later if you have one.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Palette — verified against WCAG contrast (see model_pipeline.py sibling
# scratch check; ratios: black/cream 17.1:1, black/yellow 12.6:1,
# white/black-pill 17.4:1, success-text/success-bg 10.0:1, amber-text/amber-bg
# 9.9:1, success-border/success-bg 3.1:1, amber-border/amber-bg 4.2:1 — all
# pass their respective 4.5:1 text / 3:1 UI-component thresholds).
# ---------------------------------------------------------------------------
PALETTE = {
    "bg_cream": "#FFFDF5",
    "bg_yellow": "#FFD93D",
    "bg_yellow_soft": "#FFF3C4",
    "black": "#1A1A1A",
    "white": "#FFFFFF",
    "text_muted_light": "#6B6350",   # secondary text on cream/white
    "text_muted_yellow": "#4A4326",  # secondary text on yellow (darker, same hue family)
    "success_bg": "#EAF6EC",
    "success_border": "#2F9E44",
    "success_text": "#1B4332",
    "amber_bg": "#FFF3D6",
    "amber_border": "#B85C00",
    "amber_text": "#5A3400",
}

CSS_VARS_BLOCK = ":root {\n" + "\n".join(
    f"  --{k.replace('_', '-')}: {v};" for k, v in PALETTE.items()
) + "\n}\n"


# A static, low-alpha dot grid — decorative only, no motion. See module
# docstring for why this lives here instead of in a components.html snippet.
_BACKGROUND_PATTERN_CSS = """
.stApp {
  background-color: var(--bg-cream);
  background-image:
    radial-gradient(circle, rgba(26,26,26,0.05) 1.6px, transparent 1.6px);
  background-size: 26px 26px;
  background-position: 0 0;
}
"""

BASE_STYLES = CSS_VARS_BLOCK + _BACKGROUND_PATTERN_CSS + """
html, body, [class*="css"] {
  font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* ---- headings / body text ------------------------------------------- */
h1, h2, h3 { color: var(--black); font-weight: 800; letter-spacing: -0.01em; }
p, li, label, .stMarkdown { color: var(--black); }
.subtle { color: var(--text-muted-light); }

/* ---- primary pill buttons (CTA / Submit) ------------------------------
   Real st.button(type="primary"), reskinned. :hover/transition work exactly
   the same on injected CSS as anywhere else -- the "can't do this natively"
   limitation is about Streamlit exposing no animation API, not about CSS
   itself failing to apply. */
div[data-testid="stButton"] button[kind="primary"] {
  position: relative;
  overflow: hidden;
  background: var(--black) !important;
  color: var(--white) !important;
  border: none !important;
  border-radius: 999px !important;
  padding: 0.7em 2.2em !important;
  font-weight: 700 !important;
  font-size: 1.05rem !important;
  letter-spacing: 0.01em;
  box-shadow: 0 2px 0 rgba(26,26,26,0.15);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stButton"] button[kind="primary"]::after {
  content: "";
  position: absolute;
  top: 0; left: -60%;
  width: 40%; height: 100%;
  background: linear-gradient(120deg, transparent, rgba(255,217,61,0.55), transparent);
  transform: skewX(-20deg);
  transition: left 0.55s ease;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 0 rgba(26,26,26,0.18);
}
div[data-testid="stButton"] button[kind="primary"]:hover::after { left: 120%; }
div[data-testid="stButton"] button[kind="primary"]:active { transform: translateY(0); }

/* ---- disabled "coming soon" buttons ---------------------------------- */
div[data-testid="stButton"] button:disabled {
  background: var(--bg-yellow-soft) !important;
  color: var(--text-muted-yellow) !important;
  border: 1.5px dashed var(--text-muted-yellow) !important;
  border-radius: 999px !important;
  opacity: 1 !important;
  font-weight: 600 !important;
}

/* ---- st.pills language selector --------------------------------------- */
div[data-testid="stPills"] [data-baseweb="tag"],
div[data-testid="stPills"] button {
  border-radius: 999px !important;
  border: 1.5px solid var(--black) !important;
  background: var(--white) !important;
  color: var(--black) !important;
  font-weight: 600 !important;
}
div[data-testid="stPills"] button[aria-pressed="true"] {
  background: var(--black) !important;
  color: var(--white) !important;
}

/* ---- file uploader ------------------------------------------------------ */
div[data-testid="stFileUploaderDropzone"] {
  background: var(--white) !important;
  border: 2px dashed var(--black) !important;
  border-radius: 18px !important;
}

/* ---- alert cards -------------------------------------------------------- */
.alert-card {
  border-radius: 16px;
  padding: 1.1rem 1.4rem;
  margin: 0.6rem 0 1rem 0;
  border: 1.5px solid transparent;
}
.alert-card h4 { margin: 0 0 0.35rem 0; font-size: 1.15rem; }
.alert-card p { margin: 0.15rem 0; }
.alert-card .meta { font-size: 0.92rem; color: var(--text-muted-light); }
.alert-card.success {
  background: var(--success-bg);
  border-color: var(--success-border);
  color: var(--success-text);
}
.alert-card.success h4 { color: var(--success-text); }
.alert-card.amber {
  background: var(--amber-bg);
  border-color: var(--amber-border);
  color: var(--amber-text);
}
.alert-card.amber h4 { color: var(--amber-text); }

/* ---- generic content card (PDF report, section wrappers) -------------- */
.content-card {
  background: var(--white);
  border: 1.5px solid var(--bg-yellow);
  border-radius: 18px;
  padding: 1.2rem 1.4rem;
  margin: 0.8rem 0;
}
.coming-soon-badge {
  display: inline-block;
  background: var(--bg-yellow);
  color: var(--black);
  font-size: 0.75rem;
  font-weight: 700;
  padding: 0.15em 0.7em;
  border-radius: 999px;
  margin-left: 0.5em;
  vertical-align: middle;
}
"""


def alert_card(kind: str, title: str, body_html: str, meta_html: str = "") -> str:
    """
    Static styled box for a verdict/error result. Plain HTML via st.markdown --
    no animation needed, so no reason to pay for an iframe here.
    """
    css_kind = "success" if kind == "success" else "amber"
    meta = f'<p class="meta">{meta_html}</p>' if meta_html else ""
    return (
        f'<div class="alert-card {css_kind}">'
        f"<h4>{title}</h4>{body_html}{meta}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# components.v1.html snippets — each is a complete standalone HTML document.
# ---------------------------------------------------------------------------

def landing_illustration_html() -> str:
    """
    Flat line-art seated figure reading, looping ~8s dramatization: calm
    reading -> phone rings -> panic reaction (shake, alarmed face, motion
    lines, book slips) -> worried hold -> cut back to calm.

    Choreography note: every element shares one 8s master timeline
    (`--cycle`) via matching `animation-duration`, using percentage
    keyframe offsets, so everything stays in sync without JS. A few elements
    additionally run their OWN fast, independent looping animation (idle
    bob, wave pulse, running legs) layered via a second comma-separated
    `animation` on the same element -- a fast micro-loop for continuous life,
    gated visible/invisible by the slow master-timeline animation.
    """
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{CSS_VARS_BLOCK}
html,body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
.wrap {{ display:flex; justify-content:center; align-items:center; height:100%; }}
svg {{ width:100%; max-width:320px; height:auto; display:block; }}

/* ---- idle bob (continuous, independent of the story timeline) -------- */
@keyframes idle-bob {{
  0%, 100% {{ transform: translateY(0); }}
  50%      {{ transform: translateY(-3px); }}
}}
#person {{ animation: idle-bob 2.2s ease-in-out infinite, story-shake 8s ease-in-out infinite; transform-origin: 150px 260px; }}

/* ---- master story timeline: person shake burst during panic ---------- */
@keyframes story-shake {{
  0%, 49% {{ transform: translateX(0) rotate(0); }}
  50%     {{ transform: translateX(-4px) rotate(-1.2deg); }}
  52%     {{ transform: translateX(4px) rotate(1.2deg); }}
  54%     {{ transform: translateX(-3px) rotate(-1deg); }}
  56%, 100% {{ transform: translateX(0) rotate(0); }}
}}

/* ---- face swap: calm <-> panic ---------------------------------------- */
@keyframes face-calm {{
  0%, 49.5% {{ opacity: 1; }}
  50%, 99.5% {{ opacity: 0; }}
  100% {{ opacity: 1; }}
}}
@keyframes face-panic {{
  0%, 49.5% {{ opacity: 0; }}
  50%, 99.5% {{ opacity: 1; }}
  100% {{ opacity: 0; }}
}}
#face-calm  {{ animation: face-calm 8s linear infinite; }}
#face-panic {{ animation: face-panic 8s linear infinite; }}

/* ---- book: slips down/rotates during panic, back in place after loop -- */
@keyframes book-drop {{
  0%, 49%   {{ transform: translate(0,0) rotate(0deg); }}
  58%, 90%  {{ transform: translate(-6px, 14px) rotate(-14deg); }}
  100%      {{ transform: translate(0,0) rotate(0deg); }}
}}
#book {{ animation: book-drop 8s ease-in-out infinite; transform-origin: 150px 196px; }}

/* ---- phone: rings (shake) only in its window -------------------------- */
@keyframes phone-window {{
  0%, 36%   {{ opacity: 0; }}
  37%, 50%  {{ opacity: 1; }}
  51%, 100% {{ opacity: 0; }}
}}
@keyframes phone-shake {{
  0%, 100% {{ transform: rotate(0deg); }}
  25%      {{ transform: rotate(-8deg); }}
  75%      {{ transform: rotate(8deg); }}
}}
#phone-waves {{ animation: phone-window 8s linear infinite; }}
#phone-body  {{ animation: phone-shake 0.25s ease-in-out infinite; transform-origin: 247px 139px; }}

@keyframes wave-pulse {{
  0%   {{ opacity: 0.9; transform: scale(0.6); }}
  100% {{ opacity: 0;   transform: scale(1.6); }}
}}
.wave {{ animation: wave-pulse 0.9s ease-out infinite; transform-origin: 247px 139px; }}
.wave:nth-child(2) {{ animation-delay: 0.3s; }}
.wave:nth-child(3) {{ animation-delay: 0.6s; }}

/* ---- motion lines flash during the panic burst ------------------------ */
@keyframes lines-window {{
  0%, 49%   {{ opacity: 0; }}
  51%, 63%  {{ opacity: 1; }}
  70%, 100% {{ opacity: 0; }}
}}
#motion-lines {{ animation: lines-window 8s ease-out infinite; }}
</style></head>
<body><div class="wrap">
<svg viewBox="0 0 320 320" fill="none" xmlns="http://www.w3.org/2000/svg">

  <g id="person" stroke="var(--black)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">
    <!-- crossed-leg seated base -->
    <path d="M92 268 Q150 300 208 268 Q190 250 150 250 Q110 250 92 268 Z" fill="var(--bg-yellow-soft)"/>
    <!-- torso -->
    <rect x="104" y="128" width="92" height="110" rx="28" fill="var(--white)"/>
    <!-- arms toward book -->
    <path d="M108 168 Q92 190 118 205" fill="none"/>
    <path d="M192 168 Q208 190 182 205" fill="none"/>
    <!-- head -->
    <circle cx="150" cy="90" r="30" fill="var(--bg-yellow)"/>

    <!-- calm face: looking down at the book -->
    <g id="face-calm">
      <path d="M134 90 Q140 96 146 90" fill="none" stroke-width="3.2"/>
      <path d="M154 90 Q160 96 166 90" fill="none" stroke-width="3.2"/>
      <path d="M140 106 Q150 110 160 106" fill="none" stroke-width="3"/>
    </g>

    <!-- panic face: wide eyes, raised brows, open mouth -->
    <g id="face-panic">
      <line x1="130" y1="76" x2="142" y2="80" stroke-width="3"/>
      <line x1="170" y1="76" x2="158" y2="80" stroke-width="3"/>
      <circle cx="139" cy="90" r="5.5" fill="var(--black)" stroke="none"/>
      <circle cx="161" cy="90" r="5.5" fill="var(--black)" stroke="none"/>
      <ellipse cx="150" cy="108" rx="7" ry="9" fill="var(--white)" stroke-width="3"/>
    </g>
  </g>

  <!-- book (own transform group, animated independently of #person) -->
  <g id="book" stroke="var(--black)" stroke-width="3.5" stroke-linejoin="round">
    <rect x="118" y="150" width="64" height="46" rx="5" fill="var(--white)"/>
    <line x1="150" y1="150" x2="150" y2="196"/>
  </g>

  <!-- motion lines near the head, flash during panic -->
  <g id="motion-lines" stroke="var(--black)" stroke-width="3" stroke-linecap="round">
    <line x1="104" y1="60" x2="90" y2="48"/>
    <line x1="98" y1="76" x2="80" y2="72"/>
    <line x1="196" y1="60" x2="210" y2="48"/>
    <line x1="202" y1="76" x2="220" y2="72"/>
  </g>

  <!-- phone -->
  <g id="phone">
    <g id="phone-waves" stroke="var(--black)" stroke-width="2.5" fill="none">
      <circle class="wave" cx="247" cy="139" r="22"/>
      <circle class="wave" cx="247" cy="139" r="22"/>
      <circle class="wave" cx="247" cy="139" r="22"/>
    </g>
    <rect id="phone-body" x="230" y="110" width="34" height="58" rx="8"
          fill="var(--black)" stroke="var(--black)" stroke-width="2"/>
  </g>
</svg>
</div></body></html>"""


def hamster_loader_html(label: str = "Analyzing voice…") -> str:
    """Palette-themed hamster-wheel loader, self-contained (no Python round trip)."""
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{CSS_VARS_BLOCK}
html,body {{ margin:0; padding:0; background:transparent; }}
.wrap {{ display:flex; flex-direction:column; align-items:center; justify-content:center;
         height:100%; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }}
svg {{ width:130px; height:130px; }}

@keyframes spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
#wheel {{ animation: spin 1.1s linear infinite; transform-origin: 100px 100px; }}

@keyframes bob {{ 0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-2px); }} }}
#hamster {{ animation: bob 0.28s ease-in-out infinite; }}

@keyframes leg-a {{ 0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(3px); }} }}
@keyframes leg-b {{ 0%,100% {{ transform: translateY(3px); }} 50% {{ transform: translateY(0); }} }}
.leg-front {{ animation: leg-a 0.28s ease-in-out infinite; }}
.leg-back  {{ animation: leg-b 0.28s ease-in-out infinite; }}

@keyframes pulse-text {{ 0%,100% {{ opacity: 0.6; }} 50% {{ opacity: 1; }} }}
.label {{ margin-top: 0.6rem; font-weight: 600; color: var(--black); animation: pulse-text 1.4s ease-in-out infinite; }}
</style></head>
<body><div class="wrap">
<svg viewBox="0 0 200 200">
  <g id="wheel" fill="none" stroke="var(--bg-yellow)" stroke-width="8">
    <circle cx="100" cy="100" r="70"/>
    <line x1="100" y1="30" x2="100" y2="170"/>
    <line x1="30" y1="100" x2="170" y2="100"/>
    <line x1="50" y1="50" x2="150" y2="150"/>
    <line x1="150" y1="50" x2="50" y2="150"/>
  </g>
  <g id="hamster" stroke="var(--black)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <line class="leg-front" x1="88" y1="150" x2="86" y2="162" stroke-width="4"/>
    <line class="leg-back"  x1="112" y1="150" x2="114" y2="162" stroke-width="4"/>
    <ellipse cx="100" cy="140" rx="22" ry="14" fill="var(--black)"/>
    <circle cx="122" cy="132" r="10" fill="var(--black)"/>
    <path d="M114 122 l-4 -8 l8 3 Z" fill="var(--black)"/>
    <path d="M80 142 q-10 -2 -14 4" fill="none" stroke-width="2.5"/>
  </g>
</svg>
<div class="label">{label}</div>
</div></body></html>"""


def scroll_reveal_component(
    title: str,
    body_html: str,
    badge_text: str = "Coming soon",
) -> str:
    """
    A card that fades/slides into place once the surrounding iframe scrolls
    into the OUTER page's viewport (not just its own).

    Uses `window.frameElement` from inside the iframe to read the iframe's own
    position in the parent document -- this only works because
    components.v1.html renders a same-origin `srcdoc` iframe. If that access
    is ever blocked (a stricter embedding context), the card intentionally
    starts partially visible (opacity 0.55, not 0) rather than fully hidden,
    so it degrades to "visible but not animated" instead of "invisible."
    """
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{CSS_VARS_BLOCK}
html,body {{ margin:0; padding:0; background:transparent;
             font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }}
.card {{
  background: var(--white);
  border: 1.5px solid var(--bg-yellow);
  border-radius: 18px;
  padding: 1.2rem 1.4rem;
  opacity: 0.55;
  transform: translateY(14px);
  transition: opacity 0.6s ease, transform 0.6s ease;
  box-sizing: border-box;
}}
.card.visible {{ opacity: 1; transform: translateY(0); }}
.card h4 {{ margin: 0 0 0.4rem 0; color: var(--black); font-size: 1.1rem; }}
.card p {{ margin: 0.2rem 0; color: var(--black); }}
.badge {{
  display:inline-block; background: var(--bg-yellow); color: var(--black);
  font-size: 0.75rem; font-weight: 700; padding: 0.15em 0.7em;
  border-radius: 999px; margin-left: 0.5em; vertical-align: middle;
}}
</style></head>
<body>
<div class="card" id="card">
  <h4>{title}<span class="badge">{badge_text}</span></h4>
  {body_html}
</div>
<script>
  var card = document.getElementById('card');
  function checkVisible() {{
    try {{
      var rect = window.frameElement.getBoundingClientRect();
      var vh = window.parent.innerHeight || document.documentElement.clientHeight;
      if (rect.top < vh * 0.9 && rect.bottom > 0) {{
        card.classList.add('visible');
      }}
    }} catch (e) {{
      // Cross-origin or blocked access -- fall back to the partially-visible
      // static state defined in CSS above rather than staying invisible.
      card.classList.add('visible');
    }}
  }}
  checkVisible();
  try {{ window.parent.addEventListener('scroll', checkVisible, {{ passive: true }}); }}
  catch (e) {{ /* fallback already applied above */ }}
  window.addEventListener('scroll', checkVisible, {{ passive: true }});
  setInterval(checkVisible, 500);
</script>
</body></html>"""
