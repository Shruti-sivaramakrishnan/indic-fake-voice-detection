"""Visual system and small HTML helpers for the Voice Guard Streamlit app."""

from __future__ import annotations


BASE_STYLES = r"""
:root {
  --vg-ink: #111827;
  --vg-muted: #667085;
  --vg-soft-muted: #98a2b3;
  --vg-line: #e4e7ec;
  --vg-surface: #ffffff;
  --vg-canvas: #f6f7fb;
  --vg-brand: #665cf6;
  --vg-brand-dark: #4f46d9;
  --vg-brand-soft: #efeeff;
  --vg-navy: #101828;
  --vg-success: #079455;
  --vg-success-soft: #ecfdf3;
  --vg-warning: #dc6803;
  --vg-warning-soft: #fff6ed;
  --vg-danger: #d92d20;
  --vg-danger-soft: #fff1f0;
  --vg-shadow: 0 18px 50px rgba(16, 24, 40, 0.08);
}

html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

body, .stApp { color: var(--vg-ink); }
.stApp {
  background:
    radial-gradient(circle at 9% 4%, rgba(102, 92, 246, 0.10), transparent 25rem),
    radial-gradient(circle at 93% 18%, rgba(14, 165, 233, 0.08), transparent 28rem),
    var(--vg-canvas);
}

[data-testid="stHeader"] { height: 0; background: transparent; }
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stAppViewContainer"] > .main .block-container {
  max-width: 1180px;
  padding-top: 1.35rem;
  padding-bottom: 3rem;
}

h1, h2, h3, h4 { color: var(--vg-ink); letter-spacing: -0.03em; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
  color: var(--vg-muted) !important;
}

/* Top navigation */
.vg-topbar {
  display: flex;
  align-items: center;
  min-height: 46px;
}
.vg-brand {
  display: inline-flex;
  align-items: center;
  gap: 0.72rem;
  color: var(--vg-navy);
  font-weight: 780;
  font-size: 1.08rem;
  letter-spacing: -0.02em;
}
.vg-brand-mark {
  display: grid;
  place-items: center;
  width: 38px;
  height: 38px;
  border-radius: 12px;
  color: white;
  background: linear-gradient(145deg, #7c72ff, #5147dd);
  box-shadow: 0 8px 20px rgba(102, 92, 246, 0.25);
}
.vg-brand-mark svg { width: 22px; height: 22px; }
.vg-status {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  margin-top: 0.6rem;
  color: #475467;
  font-size: 0.78rem;
  font-weight: 650;
}
.vg-status-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: #12b76a;
  box-shadow: 0 0 0 4px rgba(18, 183, 106, 0.11);
}

/* Language control */
[data-testid="stSelectbox"] { margin-top: 0.15rem; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  min-height: 42px;
  background: rgba(255,255,255,0.9) !important;
  border: 1px solid var(--vg-line) !important;
  border-radius: 12px !important;
  box-shadow: 0 3px 12px rgba(16,24,40,0.04);
}
[data-testid="stSelectbox"] [data-baseweb="select"] * { color: var(--vg-ink) !important; }

/* Hero */
.vg-hero {
  position: relative;
  overflow: hidden;
  margin: 1.15rem 0 1.6rem;
  padding: clamp(2rem, 5vw, 4.2rem);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 30px;
  color: white;
  background:
    radial-gradient(circle at 82% 15%, rgba(125, 211, 252, 0.20), transparent 19rem),
    radial-gradient(circle at 72% 110%, rgba(124, 114, 255, 0.48), transparent 24rem),
    linear-gradient(132deg, #0d1426 4%, #18213b 56%, #24204f 100%);
  box-shadow: 0 28px 70px rgba(16, 24, 40, 0.16);
}
.vg-hero::after {
  content: "";
  position: absolute;
  width: 330px;
  height: 330px;
  right: -110px;
  top: -150px;
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 50%;
  box-shadow: 0 0 0 42px rgba(255,255,255,0.025), 0 0 0 84px rgba(255,255,255,0.018);
}
.vg-eyebrow {
  position: relative;
  z-index: 1;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1.2rem;
  color: #c7d7fe;
  font-size: 0.72rem;
  font-weight: 750;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}
.vg-eyebrow::before {
  content: "";
  width: 22px;
  height: 1px;
  background: #9aa4ff;
}
.vg-hero h1 {
  position: relative;
  z-index: 1;
  max-width: 760px;
  margin: 0;
  color: white;
  font-size: clamp(2.15rem, 5.5vw, 4.3rem);
  line-height: 1.01;
  font-weight: 760;
  letter-spacing: -0.055em;
}
.vg-hero h1 span {
  color: #aaa4ff;
  background: linear-gradient(90deg, #b5b0ff, #8ed8ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.vg-hero-copy {
  position: relative;
  z-index: 1;
  max-width: 665px;
  margin: 1.35rem 0 0;
  color: #cbd5e1;
  font-size: clamp(0.98rem, 1.8vw, 1.12rem);
  line-height: 1.7;
}
.vg-chips {
  position: relative;
  z-index: 1;
  display: flex;
  flex-wrap: wrap;
  gap: 0.62rem;
  margin-top: 1.55rem;
}
.vg-chip {
  padding: 0.48rem 0.78rem;
  color: #e6eafd;
  font-size: 0.77rem;
  font-weight: 620;
  border: 1px solid rgba(255,255,255,0.13);
  border-radius: 999px;
  background: rgba(255,255,255,0.055);
  backdrop-filter: blur(8px);
}

/* Section and workspace cards */
.vg-section-kicker {
  margin-bottom: 0.35rem;
  color: var(--vg-brand-dark);
  font-size: 0.72rem;
  font-weight: 780;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.vg-section-title {
  margin: 0 0 0.35rem;
  font-size: clamp(1.45rem, 2.5vw, 2rem);
  font-weight: 735;
}
.vg-section-copy { margin: 0 0 1.1rem; color: var(--vg-muted); line-height: 1.62; }

[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255,255,255,0.94);
  border: 1px solid var(--vg-line) !important;
  border-radius: 22px !important;
  box-shadow: 0 14px 40px rgba(16,24,40,0.055);
}
[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 0.35rem; }

[data-testid="stFileUploader"] section {
  min-height: 180px;
  padding: 1.5rem 1rem;
  border: 1.5px dashed #b9b5f7 !important;
  border-radius: 18px !important;
  background: linear-gradient(145deg, #fbfbff, #f6f5ff) !important;
  transition: border-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
}
[data-testid="stFileUploader"] section:hover {
  border-color: var(--vg-brand) !important;
  box-shadow: 0 10px 28px rgba(102,92,246,0.10);
  transform: translateY(-1px);
}
[data-testid="stFileUploader"] button {
  border: 1px solid #d3d0ff !important;
  border-radius: 10px !important;
  color: var(--vg-brand-dark) !important;
  background: white !important;
  font-weight: 700 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--vg-muted) !important; }

.vg-file-meta {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1rem;
  align-items: center;
  margin: 0.75rem 0 0.9rem;
  padding: 0.85rem 1rem;
  border: 1px solid var(--vg-line);
  border-radius: 14px;
  background: #fafbfc;
}
.vg-file-name {
  overflow: hidden;
  color: var(--vg-ink);
  font-size: 0.88rem;
  font-weight: 680;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.vg-file-size { color: var(--vg-muted); font-size: 0.8rem; font-weight: 600; }

[data-testid="stAudio"] { margin-top: 0.5rem; }
[data-testid="stAudio"] audio { width: 100%; }

[data-testid="stButton"] button[kind="primary"] {
  min-height: 51px;
  border: 0 !important;
  border-radius: 13px !important;
  color: white !important;
  background: linear-gradient(135deg, #6d63f7, #5147d9) !important;
  box-shadow: 0 9px 22px rgba(102,92,246,0.24);
  font-size: 0.96rem !important;
  font-weight: 730 !important;
  transition: transform 150ms ease, box-shadow 150ms ease, filter 150ms ease;
}
[data-testid="stButton"] button[kind="primary"] * { color: white !important; }
[data-testid="stButton"] button[kind="primary"]:hover:not(:disabled) {
  filter: brightness(1.04);
  transform: translateY(-1px);
  box-shadow: 0 13px 28px rgba(102,92,246,0.30);
}
[data-testid="stButton"] button:disabled { box-shadow: none !important; opacity: 0.48; }

.vg-process-card {
  height: 100%;
  padding: 1.45rem;
  border: 1px solid var(--vg-line);
  border-radius: 22px;
  background: rgba(255,255,255,0.9);
  box-shadow: 0 14px 40px rgba(16,24,40,0.045);
}
.vg-process-card h3 { margin: 0 0 0.35rem; font-size: 1.05rem; letter-spacing: -0.02em; }
.vg-process-intro { margin: 0 0 1.25rem; color: var(--vg-muted); font-size: 0.86rem; line-height: 1.55; }
.vg-step {
  display: grid;
  grid-template-columns: 34px 1fr;
  gap: 0.8rem;
  position: relative;
  padding-bottom: 1.08rem;
}
.vg-step:not(:last-child)::after {
  content: "";
  position: absolute;
  left: 16px;
  top: 34px;
  bottom: 2px;
  width: 1px;
  background: #e6e7f2;
}
.vg-step-index {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border: 1px solid #d8d6ff;
  border-radius: 11px;
  color: var(--vg-brand-dark);
  background: var(--vg-brand-soft);
  font-size: 0.76rem;
  font-weight: 780;
}
.vg-step strong { display: block; padding-top: 0.05rem; font-size: 0.88rem; }
.vg-step p { margin: 0.2rem 0 0; color: var(--vg-muted); font-size: 0.79rem; line-height: 1.48; }
.vg-privacy-strip {
  margin-top: 0.35rem;
  padding: 0.85rem 0.9rem;
  border-radius: 13px;
  color: #344054;
  background: #f8f9fc;
  font-size: 0.77rem;
  line-height: 1.5;
}

/* Result */
.vg-result-shell {
  --result-accent: var(--vg-warning);
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1.5rem;
  align-items: center;
  margin-top: 1.2rem;
  padding: clamp(1.35rem, 3vw, 2rem);
  border: 1px solid var(--vg-line);
  border-top: 4px solid var(--result-accent);
  border-radius: 22px;
  background: white;
  box-shadow: var(--vg-shadow);
}
.vg-result-shell.real { --result-accent: var(--vg-success); }
.vg-result-shell.fake { --result-accent: var(--vg-danger); }
.vg-result-shell.uncertain { --result-accent: var(--vg-warning); }
.vg-result-label {
  color: var(--result-accent);
  font-size: 0.7rem;
  font-weight: 790;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.vg-result-shell h2 { margin: 0.4rem 0 0.5rem; font-size: clamp(1.7rem, 3.3vw, 2.5rem); }
.vg-result-shell p { max-width: 650px; margin: 0; color: var(--vg-muted); line-height: 1.6; }
.vg-score-ring {
  display: grid;
  place-items: center;
  width: 132px;
  height: 132px;
  padding: 10px;
  border-radius: 50%;
  background: conic-gradient(var(--result-accent) calc(var(--score) * 1%), #eef0f5 0);
}
.vg-score-ring-inner {
  display: grid;
  place-content: center;
  width: 100%;
  height: 100%;
  border-radius: 50%;
  text-align: center;
  background: white;
  box-shadow: inset 0 0 0 1px #f0f1f4;
}
.vg-score-ring strong { color: var(--vg-ink); font-size: 1.45rem; line-height: 1; }
.vg-score-ring span { margin-top: 0.32rem; color: var(--vg-muted); font-size: 0.64rem; font-weight: 700; text-transform: uppercase; }
.vg-score-note { margin: 0.8rem 0 0 !important; color: var(--vg-muted); font-size: 0.78rem; }

.vg-scale-card {
  margin: 0.85rem 0 1rem;
  padding: 1rem 1.15rem;
  border: 1px solid var(--vg-line);
  border-radius: 16px;
  background: white;
}
.vg-scale-track {
  position: relative;
  height: 8px;
  overflow: hidden;
  border-radius: 999px;
  background: linear-gradient(90deg, #8ee0bd 0%, #e9ecf2 50%, #ffaaa4 100%);
}
.vg-scale-marker {
  position: absolute;
  top: 50%;
  left: calc(var(--score) * 1%);
  width: 16px;
  height: 16px;
  border: 3px solid white;
  border-radius: 50%;
  background: var(--vg-navy);
  box-shadow: 0 2px 7px rgba(16,24,40,0.28);
  transform: translate(-50%, -50%);
}
.vg-scale-labels { display: flex; justify-content: space-between; margin-top: 0.58rem; color: var(--vg-muted); font-size: 0.72rem; font-weight: 600; }

.alert-card {
  margin: 0.75rem 0;
  padding: 1.05rem 1.15rem;
  border: 1px solid #fedf89;
  border-radius: 16px;
  color: #7a2e0e;
  background: var(--vg-warning-soft);
}
.alert-card.success { border-color: #abefc6; color: #05603a; background: var(--vg-success-soft); }
.alert-card.danger { border-color: #fecdca; color: #912018; background: var(--vg-danger-soft); }
.alert-card h4 { margin: 0 0 0.3rem; color: inherit; font-size: 0.94rem; letter-spacing: -0.01em; }
.alert-card p { margin: 0; color: inherit; font-size: 0.84rem; line-height: 1.55; }
.alert-card .meta { margin-top: 0.38rem; font-weight: 650; }

.vg-info-card {
  height: 100%;
  padding: 1.1rem 1.15rem;
  border: 1px solid var(--vg-line);
  border-radius: 16px;
  background: white;
}
.vg-info-card .vg-info-icon {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  margin-bottom: 0.8rem;
  border-radius: 10px;
  color: var(--vg-brand-dark);
  background: var(--vg-brand-soft);
  font-size: 0.92rem;
}
.vg-info-card h4 { margin: 0 0 0.35rem; font-size: 0.9rem; }
.vg-info-card p { margin: 0; color: var(--vg-muted); font-size: 0.78rem; line-height: 1.5; }

[data-testid="stDownloadButton"] button {
  min-height: 48px;
  border: 1px solid #d3d0ff !important;
  border-radius: 12px !important;
  color: var(--vg-brand-dark) !important;
  background: #f8f7ff !important;
  font-weight: 720 !important;
}
[data-testid="stDownloadButton"] button:hover {
  border-color: var(--vg-brand) !important;
  background: var(--vg-brand-soft) !important;
}
[data-testid="stExpander"] {
  border: 1px solid var(--vg-line) !important;
  border-radius: 14px !important;
  background: rgba(255,255,255,0.8) !important;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--vg-ink) !important; }

.vg-footer {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  margin-top: 2.3rem;
  padding-top: 1.2rem;
  border-top: 1px solid var(--vg-line);
  color: var(--vg-muted);
  font-size: 0.74rem;
  line-height: 1.5;
}
.vg-footer strong { color: #344054; }

@media (max-width: 700px) {
  [data-testid="stAppViewContainer"] > .main .block-container { padding: 0.8rem 1rem 2rem; }
  .vg-hero { padding: 2rem 1.35rem; border-radius: 22px; }
  .vg-hero h1 { font-size: 2.25rem; }
  .vg-result-shell { grid-template-columns: 1fr; }
  .vg-score-ring { width: 116px; height: 116px; }
  .vg-footer { flex-direction: column; }
}
"""


def alert_card(kind: str, title: str, body_html: str, meta_html: str = "") -> str:
    """Return a semantic alert card. Callers must escape user-controlled text."""
    css_kind = kind if kind in {"success", "danger"} else "warning"
    meta = f'<p class="meta">{meta_html}</p>' if meta_html else ""
    return (
        f'<div class="alert-card {css_kind}">'
        f"<h4>{title}</h4>{body_html}{meta}"
        "</div>"
    )
