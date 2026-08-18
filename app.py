"""
2-screen Streamlit demo for the Indic fake-voice detector.

Run:
    pip install streamlit
    streamlit run app.py

Screen 1 (landing) dramatizes the WhatsApp panic-call scam scenario. Screen 2
lets a user upload a clip and calls model_pipeline.analyze_bytes() for real
inference -- no mock data anywhere in the result path.

Design notes (see ui_kit.py for the fuller rationale):
  * Palette + static widget theming (pill buttons, st.pills, alert cards) is
    injected as CSS into the main document -- the only way to theme
    Streamlit's real, interactive widgets.
  * The landing illustration/dramatization, the hamster-wheel loader, and the
    WhatsApp scroll-reveal card are self-contained HTML/CSS/JS rendered via
    st.components.v1.html, since they need real keyframe animation that no
    native Streamlit widget provides.
  * The language selector is a UI element only -- it is intentionally NOT
    passed into analyze_bytes() as a language_hint. There is no language
    classifier in the pipeline, and the spec calls for detected_language to
    show "unknown" as-is; wiring the selector into the hint would make that
    field lie. A caption next to the selector says so explicitly.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

import ui_kit

MODEL_IMPORT_ERROR: str | None = None
try:
    import model_pipeline as mp
except Exception as exc:  # noqa: BLE001 -- surfaced as an amber card, not a crash
    mp = None
    MODEL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


st.set_page_config(
    page_title="Voice Guard — Fake Voice Detector",
    page_icon="🛡️",
    layout="centered",
)
st.markdown(f"<style>{ui_kit.BASE_STYLES}</style>", unsafe_allow_html=True)

st.session_state.setdefault("screen", "landing")
st.session_state.setdefault("result", None)
st.session_state.setdefault("result_error", None)
st.session_state.setdefault("ui_language", "English")

LANGUAGES = ["English", "Hindi", "Tamil", "Kannada"]
ACCEPTED_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]


def go_to(screen: str) -> None:
    st.session_state.screen = screen


# ============================================================================
# SCREEN 1 — Landing
# ============================================================================

def render_landing() -> None:
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0;'>🛡️ Voice Guard</h1>"
        "<p class='subtle' style='text-align:center; font-size:1.05rem;'>"
        "AI-generated voice detection for the calls that try to catch you off guard."
        "</p>",
        unsafe_allow_html=True,
    )

    components.html(ui_kit.landing_illustration_html(), height=300, scrolling=False)

    st.markdown(
        """
        <div class="content-card" style="text-align:center;">
          <p style="margin:0;">
            <strong>It starts with a call.</strong> A voice that sounds exactly like
            someone you trust — a relative, a friend — asking for money, urgently,
            right now. It's a cloned voice built from a few seconds of audio.
          </p>
          <p style="margin-top:0.6rem;">
            Voice Guard checks a recording in seconds and tells you whether it's
            real or AI-generated, before you act on it.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("Check a voice message →", type="primary", use_container_width=True):
            go_to("analyze")
            st.rerun()


# ============================================================================
# SCREEN 2 — Upload & Analyze
# ============================================================================

def run_analysis(audio_bytes: bytes):
    """Wrap analyze_bytes so every failure mode lands as an amber card, never
    an unhandled Streamlit traceback."""
    if MODEL_IMPORT_ERROR:
        raise RuntimeError(
            f"Model pipeline could not be loaded ({MODEL_IMPORT_ERROR}). "
            "This demo needs model_pipeline.py's dependencies (torch, "
            "transformers, etc.) installed and checkpoints/best_model.pt present."
        )
    # Deliberately no language_hint -- see module docstring.
    return mp.analyze_bytes(audio_bytes)


def render_result_card(result) -> None:
    verdict = result.verdict
    confidence_pct = f"{result.confidence * 100:.1f}%"
    lang_eer = (
        f"{result.language_eer * 100:.2f}%"
        if result.language_eer is not None
        else "not available yet"
    )
    meta = (
        f"Confidence: {confidence_pct} &nbsp;•&nbsp; "
        f"Detected language: {result.detected_language} &nbsp;•&nbsp; "
        f"Language EER: {lang_eer}"
    )

    if verdict == "real":
        html = ui_kit.alert_card(
            "success", "✅ Likely Real",
            "<p>This clip matches the acoustic pattern of a genuine human voice.</p>",
            meta,
        )
    elif verdict == "fake":
        html = ui_kit.alert_card(
            "amber", "🚨 Likely Fake — AI-generated",
            "<p>This clip shows patterns consistent with synthetic speech. "
            "Treat any request made in it with caution.</p>",
            meta,
        )
    elif verdict == "uncertain":
        html = ui_kit.alert_card(
            "amber", "⚠️ Uncertain — inconclusive",
            "<p>The model couldn't confidently call this real or fake. "
            "Don't rely on this result alone — verify the caller another way.</p>",
            meta,
        )
    else:  # defensive fallback for any future verdict value
        html = ui_kit.alert_card(
            "amber", "⚠️ Unrecognized result",
            f"<p>Got an unexpected verdict value: <code>{verdict}</code>.</p>",
            meta,
        )
    st.markdown(html, unsafe_allow_html=True)

    with st.expander("Technical details"):
        st.write(f"**Raw score (uncalibrated logit):** {result.raw_score:.4f}")
        st.write(f"**Audio hash (sha256):** `{result.audio_hash}`")
        st.json(result.metadata)


def render_analyze() -> None:
    if st.button("← Back", key="back_btn"):
        go_to("landing")
        st.rerun()

    st.markdown("<h2>Upload a voice clip</h2>", unsafe_allow_html=True)

    st.markdown(
        "<p class='subtle' style='margin-bottom:0.3rem;'>Interface language</p>",
        unsafe_allow_html=True,
    )
    st.pills(
        "Interface language",
        LANGUAGES,
        default=st.session_state.ui_language,
        key="ui_language",
        label_visibility="collapsed",
    )
    st.caption(
        "Full interface translation is coming soon — this selector doesn't "
        "change analysis. It also isn't passed to the model: there is no "
        "language classifier yet, so **detected language always reads "
        "\"unknown\"** below, honestly."
    )

    uploaded = st.file_uploader(
        "Upload an audio clip", type=ACCEPTED_TYPES, label_visibility="collapsed"
    )

    submit = st.button("Analyze this clip", type="primary", use_container_width=True)

    if submit:
        if uploaded is None:
            st.markdown(
                ui_kit.alert_card(
                    "amber", "⚠️ No file selected",
                    "<p>Upload an audio clip above before analyzing.</p>",
                ),
                unsafe_allow_html=True,
            )
        else:
            loader_slot = st.empty()
            with loader_slot:
                components.html(ui_kit.hamster_loader_html(), height=210, scrolling=False)
            try:
                audio_bytes = uploaded.getvalue()
                result = run_analysis(audio_bytes)
                st.session_state.result = result
                st.session_state.result_error = None
            except Exception as exc:  # noqa: BLE001 -- shown as an amber card
                st.session_state.result = None
                st.session_state.result_error = str(exc)
            loader_slot.empty()

    if st.session_state.result_error:
        st.markdown(
            ui_kit.alert_card(
                "amber", "⚠️ Something went wrong",
                f"<p>{st.session_state.result_error}</p>",
            ),
            unsafe_allow_html=True,
        )
    elif st.session_state.result is not None:
        render_result_card(st.session_state.result)

    st.markdown("<h3 style='margin-top:2rem;'>PDF Evidence Report</h3>", unsafe_allow_html=True)
    st.markdown(
        '<div class="content-card">'
        "<p style='margin:0;'>Generate a shareable PDF with the verdict, "
        "confidence, and technical details for this clip."
        '<span class="coming-soon-badge">Coming soon</span></p>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.button("📄 Generate PDF Evidence Report", disabled=True)

    # Pushes the WhatsApp section below the fold so "appears on scroll" is real.
    st.markdown("<div style='height:14vh;'></div>", unsafe_allow_html=True)

    st.markdown("<h3>Report scam calls directly</h3>", unsafe_allow_html=True)
    components.html(
        ui_kit.scroll_reveal_component(
            "WhatsApp Bot",
            "<p>Forward a suspicious voice note straight to our WhatsApp "
            "number and get a verdict back in seconds — no need to open "
            "this page.</p>",
        ),
        height=170,
        scrolling=False,
    )


# ============================================================================

if st.session_state.screen == "landing":
    render_landing()
else:
    render_analyze()
