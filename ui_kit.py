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

/* ============================================================================
   TEXT-COLOR INVARIANT: dark bg -> light text, light bg -> dark text, no
   exceptions, enforced PER COMPONENT (not one global rule).

   Why per-component: Streamlit wraps widget labels in nested markdown
   containers (e.g. a button's visible text sits in a `<p>` several levels
   inside the `<button>`). A single blanket rule like `p { color: black }`
   directly matches THAT inner `<p>` too and overrides the button's correctly
   INHERITED white text -- confirmed via a real rendered-DOM audit
   (Playwright + Chromium): a `button[kind="primary"]` correctly had
   `color: white` on itself, but its inner `<p>` showed `color: rgb(26,26,26)`
   (black-on-black-background, i.e. invisible), entirely because of that one
   overbroad rule. There is no blanket-selector fix for this: Streamlit reuses
   the same markdown-wrapper structure everywhere, inside buttons and outside
   them, so nothing about the WRAPPER distinguishes context.

   The fix: set a single default at the top (inherited downward, not
   re-matched by a bare tag selector), then for every component that needs a
   DIFFERENT color, force it with `!important` on the component root AND
   `* ` (all descendants) -- so it wins regardless of how many wrapper levels
   Streamlit inserts.

   `.streamlit/config.toml` also forces Streamlit's own theme to light, which
   fixes a SEPARATE bug: several native widget containers (file uploader,
   selectbox, expander) paint themselves with Streamlit's own dark-theme
   background when the visitor's OS/browser prefers dark, while this CSS
   assumes a light page -- also confirmed via the same audit, in dark
   color-scheme specifically. Both fixes are needed; they cover different
   failure modes.
   ============================================================================ */

/* ---- default: set once, inherited down, never re-matched by a bare tag */
body, .stApp { color: var(--black); }
h1, h2, h3, h4 { color: var(--black); font-weight: 800; letter-spacing: -0.01em; }
.subtle { color: var(--text-muted-light); }

/* Streamlit sets its own explicit color on these; force ours on every
   descendant so it isn't left in a theme-dependent state. */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
  color: var(--text-muted-light) !important;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
  color: var(--black) !important;
}
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {
  color: var(--black) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] * {
  color: var(--text-muted-light) !important;
}

/* ---- primary pill buttons (CTA / Submit) ------------------------------
   Real st.button(type="primary"), reskinned. :hover/transition work exactly
   the same on injected CSS as anywhere else -- the "can't do this natively"
   limitation is about Streamlit exposing no animation API, not about CSS
   itself failing to apply. */
[data-testid="stButton"] button[kind="primary"],
[data-testid="stButton"] button[kind="primary"] * {
  color: var(--white) !important;
}
[data-testid="stButton"] button[kind="primary"] {
  position: relative;
  overflow: hidden;
  background: var(--black) !important;
  border: none !important;
  border-radius: 999px !important;
  padding: 0.7em 2.2em !important;
  font-weight: 700 !important;
  font-size: 1.05rem !important;
  letter-spacing: 0.01em;
  box-shadow: 0 2px 0 rgba(26,26,26,0.15);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stButton"] button[kind="primary"]::after {
  content: "";
  position: absolute;
  top: 0; left: -60%;
  width: 40%; height: 100%;
  background: linear-gradient(120deg, transparent, rgba(255,217,61,0.55), transparent);
  transform: skewX(-20deg);
  transition: left 0.55s ease;
}
[data-testid="stButton"] button[kind="primary"]:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 0 rgba(26,26,26,0.18);
}
[data-testid="stButton"] button[kind="primary"]:hover::after { left: 120%; }
[data-testid="stButton"] button[kind="primary"]:active { transform: translateY(0); }

/* ---- tertiary buttons: the small circular arrow-nav controls ---------- */
[data-testid="stButton"] button[kind="tertiary"],
[data-testid="stButton"] button[kind="tertiary"] * {
  color: var(--black) !important;
}
[data-testid="stButton"] button[kind="tertiary"] {
  background: var(--white) !important;
  border: 2px solid var(--black) !important;
  border-radius: 999px !important;
  min-width: 2.6em !important;
  height: 2.6em !important;
  padding: 0 !important;
  font-size: 1.25rem !important;
  font-weight: 700 !important;
  line-height: 1 !important;
  transition: background 0.15s ease, transform 0.15s ease;
}
[data-testid="stButton"] button[kind="tertiary"]:hover:not(:disabled) {
  background: var(--bg-yellow) !important;
  transform: translateY(-1px);
}
[data-testid="stButton"] button[kind="tertiary"]:disabled,
[data-testid="stButton"] button[kind="tertiary"]:disabled * {
  color: var(--text-muted-yellow) !important;
  opacity: 1 !important;
}
[data-testid="stButton"] button[kind="tertiary"]:disabled {
  background: var(--bg-cream) !important;
  border-color: var(--text-muted-yellow) !important;
}

/* ---- disabled "coming soon" buttons ------------------------------------ */
[data-testid="stButton"] button[kind="secondary"]:disabled,
[data-testid="stButton"] button[kind="secondary"]:disabled * {
  color: var(--text-muted-yellow) !important;
}
[data-testid="stButton"] button[kind="secondary"]:disabled {
  background: var(--bg-yellow-soft) !important;
  border: 1.5px dashed var(--text-muted-yellow) !important;
  border-radius: 999px !important;
  opacity: 1 !important;
  font-weight: 600 !important;
}

/* ---- language dropdown (st.selectbox), top-right ----------------------- */
[data-testid="stSelectbox"] label, [data-testid="stSelectbox"] label * {
  color: var(--black) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] * {
  color: var(--black) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: var(--white) !important;
  border: 1.5px solid var(--black) !important;
  border-radius: 999px !important;
}
/* the open dropdown menu is portaled to <body>, styled separately below */
ul[data-testid="stSelectboxVirtualDropdown"],
ul[data-testid="stSelectboxVirtualDropdown"] * {
  color: var(--black) !important;
  background: var(--white) !important;
}
li[data-testid="stSelectboxVirtualDropdown"]:hover,
ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
  background: var(--bg-yellow-soft) !important;
}

/* ---- file uploader ------------------------------------------------------
   Note: the dropzone is a <section>, not a <div> -- an earlier version of
   this rule was scoped to `div[data-testid=...]` and silently never matched
   anything. Bare `[data-testid=...]` (no tag prefix) is used throughout this
   file now specifically to avoid repeating that mistake. */
[data-testid="stFileUploaderDropzone"] {
  background: var(--white) !important;
  border: 2px dashed var(--black) !important;
  border-radius: 18px !important;
}
[data-testid="stFileUploaderDropzone"] button[kind="secondary"] {
  background: var(--bg-yellow) !important;
  border: none !important;
}
[data-testid="stFileUploaderDropzone"] button[kind="secondary"],
[data-testid="stFileUploaderDropzone"] button[kind="secondary"] * {
  color: var(--black) !important;
}

/* ---- alert cards --------------------------------------------------------
   Own static HTML (authored directly, not Streamlit-wrapped) but still
   forced with !important + descendant selector -- cheap insurance against
   any future global rule (including this file's own) reaching in. */
.alert-card {
  border-radius: 16px;
  padding: 1.1rem 1.4rem;
  margin: 0.6rem 0 1rem 0;
  border: 1.5px solid transparent;
}
.alert-card h4 { margin: 0 0 0.35rem 0; font-size: 1.15rem; }
.alert-card p { margin: 0.15rem 0; }
.alert-card .meta { font-size: 0.92rem; }
.alert-card.success, .alert-card.success * {
  color: var(--success-text) !important;
}
.alert-card.success { background: var(--success-bg); border-color: var(--success-border); }
.alert-card.amber, .alert-card.amber * {
  color: var(--amber-text) !important;
}
.alert-card.amber { background: var(--amber-bg); border-color: var(--amber-border); }

/* ---- generic content card (PDF report, section wrappers) --------------- */
.content-card, .content-card * { color: var(--black) !important; }
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
  color: var(--black) !important;
  font-size: 0.75rem;
  font-weight: 700;
  padding: 0.15em 0.7em;
  border-radius: 999px;
  margin-left: 0.5em;
  vertical-align: middle;
}

/* ---- header row: back/next arrows + language dropdown ------------------ */
.nav-spacer { height: 0.2rem; }
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


def whatsapp_icon_svg(size: int = 56) -> str:
    """
    Generic chat-bubble + phone glyph, flat line-art, consistent with the
    illustration style used elsewhere. Deliberately not a literal WhatsApp
    logo -- a shape that reads as "messaging", nothing brand-specific.
    """
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none"
      xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <circle cx="32" cy="32" r="30" fill="var(--bg-yellow-soft)"/>
  <path d="M20 22h20a4 4 0 0 1 4 4v10a4 4 0 0 1-4 4H28l-8 6v-6a4 4 0 0 1-4-4V26a4 4 0 0 1 4-4Z"
        fill="var(--white)" stroke="var(--black)" stroke-width="3" stroke-linejoin="round"/>
  <circle cx="26" cy="30" r="2" fill="var(--black)"/>
  <circle cx="32" cy="30" r="2" fill="var(--black)"/>
  <circle cx="38" cy="30" r="2" fill="var(--black)"/>
</svg>"""


def scroll_reveal_component(
    title: str,
    body_html: str,
    badge_text: str = "Coming soon",
    icon_svg: str = "",
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
    icon_html = f'<div class="icon">{icon_svg}</div>' if icon_svg else ""
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
  display: flex;
  align-items: flex-start;
  gap: 0.9rem;
}}
.card.visible {{ opacity: 1; transform: translateY(0); }}
.card .icon {{ flex-shrink: 0; }}
.card .text {{ flex: 1; min-width: 0; }}
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
  {icon_html}
  <div class="text">
    <h4>{title}<span class="badge">{badge_text}</span></h4>
    {body_html}
  </div>
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


def live_call_companion_html() -> str:
    """
    Flat line-art figure waiting mid-call: idle stand -> checks a wrist watch
    -> shoulders drop in a sigh, with a small breath-wisp cue -> settles ->
    loops. Same synced-master-timeline technique as landing_illustration_html
    (all elements share one animation-duration with percentage keyframe
    offsets), just a shorter ~5s cycle since this is one continuous mood
    rather than a multi-beat story.
    """
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{CSS_VARS_BLOCK}
html,body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
.wrap {{ display:flex; justify-content:center; align-items:center; height:100%; }}
svg {{ width:100%; max-width:220px; height:auto; display:block; }}

@keyframes idle-bob {{
  0%, 100% {{ transform: translateY(0); }}
  50%      {{ transform: translateY(-2px); }}
}}
#figure {{ animation: idle-bob 2.4s ease-in-out infinite; transform-origin: 110px 230px; }}

/* master timeline (shared by arm-raise / shoulder-slump / sigh-window below):
   0-40% idle wait, 40-58% arm raises to check watch, 58-72% shoulders drop /
   sigh, 72-100% settle back to idle */
@keyframes arm-raise {{
  0%, 38%   {{ transform: rotate(0deg); }}
  46%, 56%  {{ transform: rotate(-58deg); }}
  64%, 100% {{ transform: rotate(0deg); }}
}}
#arm-watch {{ animation: arm-raise 5s ease-in-out infinite; transform-origin: 130px 168px; }}

@keyframes shoulder-slump {{
  0%, 56%   {{ transform: translateY(0); }}
  66%, 88%  {{ transform: translateY(4px); }}
  100%      {{ transform: translateY(0); }}
}}
#shoulders {{ animation: shoulder-slump 5s ease-in-out infinite; }}

@keyframes sigh-window {{
  0%, 60%   {{ opacity: 0; transform: translateY(0); }}
  66%       {{ opacity: 0.9; transform: translateY(-2px); }}
  85%       {{ opacity: 0; transform: translateY(-14px); }}
  100%      {{ opacity: 0; transform: translateY(0); }}
}}
#sigh {{ animation: sigh-window 5s ease-out infinite; }}
</style></head>
<body><div class="wrap">
<svg viewBox="0 0 220 260" fill="none" xmlns="http://www.w3.org/2000/svg">

  <g id="figure" stroke="var(--black)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">
    <!-- legs / base -->
    <path d="M92 230 L88 200 M128 230 L132 200" fill="none"/>
    <!-- torso -->
    <g id="shoulders">
      <rect x="86" y="130" width="48" height="80" rx="20" fill="var(--bg-yellow-soft)"/>
    </g>
    <!-- static arm, resting -->
    <path d="M92 140 Q78 165 84 190" fill="none"/>
    <!-- arm that raises to check the watch -->
    <g id="arm-watch">
      <path d="M128 140 Q142 155 132 168" fill="none"/>
      <circle cx="132" cy="168" r="6" fill="var(--white)" stroke="var(--black)" stroke-width="2.5"/>
      <line x1="132" y1="164" x2="132" y2="168" stroke-width="1.8"/>
      <line x1="132" y1="168" x2="135" y2="168" stroke-width="1.8"/>
    </g>
    <!-- head -->
    <circle cx="110" cy="104" r="26" fill="var(--bg-yellow)"/>
    <!-- calm-tired face -->
    <path d="M98 104 Q103 108 108 104" fill="none" stroke-width="3"/>
    <path d="M112 104 Q117 108 122 104" fill="none" stroke-width="3"/>
    <path d="M100 118 Q110 114 120 118" fill="none" stroke-width="3"/>
  </g>

  <!-- breath / sigh wisp -->
  <g id="sigh" stroke="var(--black)" stroke-width="2.5" fill="none" stroke-linecap="round">
    <path d="M132 96 Q140 90 136 82 Q132 76 138 70"/>
  </g>
</svg>
</div></body></html>"""
