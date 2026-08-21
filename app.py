"""
2-screen Streamlit demo for the Indic fake-voice detector.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Screen 1 (landing) dramatizes the WhatsApp panic-call scam scenario. Screen 2
lets a user upload a clip and calls model_pipeline.analyze_bytes() for real
inference -- no mock data anywhere in the result path.

Design notes (see ui_kit.py for the fuller rationale):
  * Palette + static widget theming (pill buttons, alert cards, selectbox) is
    injected as CSS into the main document -- the only way to theme
    Streamlit's real, interactive widgets. .streamlit/config.toml forces
    Streamlit's OWN theme to light regardless of the visitor's OS/browser
    preference, since several native widgets otherwise paint themselves with
    Streamlit's dark-theme background while this CSS assumes a light page --
    confirmed as a real bug via a rendered-DOM audit, not assumed.
  * The text-color invariant (dark bg -> light text, light bg -> dark text)
    is enforced PER COMPONENT in ui_kit.BASE_STYLES, not by one global rule --
    a blanket `p { color: black }` was found, by the same audit, to reach
    inside a button and override its correctly-inherited white text. See the
    long comment at the top of BASE_STYLES for the full explanation.
  * The landing illustration/dramatization, the hamster-wheel loader, the
    Live Call Companion illustration, and the WhatsApp/PDF scroll-reveal
    cards are self-contained HTML/CSS/JS rendered via st.components.v1.html,
    since they need real keyframe animation no native Streamlit widget
    provides.
  * The language selector is a UI element only -- it is intentionally NOT
    passed into analyze_bytes() as a language_hint. There is no language
    classifier in the pipeline, and the spec calls for detected_language to
    show "unknown" as-is; wiring the selector into the hint would make that
    field lie. The caption next to the selector says so explicitly, in
    whichever language is currently selected.
"""

from __future__ import annotations

import truststore
truststore.inject_into_ssl()

import evidence_report

import streamlit as st
import streamlit.components.v1 as components

import i18n
import ui_kit
from i18n import t

MODEL_IMPORT_ERROR: str | None = None

try:
    import model_pipeline as mp
except Exception as exc:  # noqa: BLE001 -- surfaced as an amber card, not a crash
    mp = None
    MODEL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


st.set_page_config(
    page_title="Voice Guard",
    page_icon="🛡️",
    layout="centered",
)

st.markdown(
    f"<style>{ui_kit.BASE_STYLES}</style>",
    unsafe_allow_html=True,
)

st.session_state.setdefault("screen", "landing")
st.session_state.setdefault("result", None)
st.session_state.setdefault("result_error", None)
st.session_state.setdefault("ui_language", "English")

ACCEPTED_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]


def go_to(screen: str) -> None:
    st.session_state.screen = screen


def current_lang() -> str:
    return st.session_state.get("ui_language", "English")


def render_header(show_back: bool, show_next: bool) -> None:
    """Persistent arrow-nav (prev/next between the app's two screens) plus
    the language dropdown, top-right, on every screen."""
    lang = current_lang()
    col_back, _spacer, col_next, col_lang = st.columns([1, 4.4, 1, 2.2])

    with col_back:
        if st.button(
            "←",
            type="tertiary",
            disabled=not show_back,
            help=t("back_arrow_help", lang),
            key="nav_back",
        ):
            go_to("landing")
            st.rerun()

    with col_next:
        if st.button(
            "→",
            type="tertiary",
            disabled=not show_next,
            help=t("next_arrow_help", lang),
            key="nav_next",
        ):
            go_to("analyze")
            st.rerun()

    with col_lang:
        st.selectbox(
            t("interface_language_label", lang),
            i18n.LANGUAGES,
            key="ui_language",
            label_visibility="collapsed",
        )


# ============================================================================
# SCREEN 1 -- Landing
# ============================================================================

def render_landing() -> None:
    lang = current_lang()
    render_header(show_back=False, show_next=True)

    st.markdown(
        f"<h1 style='text-align:center; margin-bottom:0;'>🛡️ "
        f"{t('app_title', lang)}</h1>"
        f"<p class='subtle' style='text-align:center; font-size:1.05rem;'>"
        f"{t('app_tagline', lang)}</p>",
        unsafe_allow_html=True,
    )

    components.html(
        ui_kit.landing_illustration_html(),
        height=300,
        scrolling=False,
    )

    st.markdown(
        f"""
        <div class="content-card" style="text-align:center;">
          <p style="margin:0;">{t('landing_p1', lang)}</p>
          <p style="margin-top:0.6rem;">{t('landing_p2', lang)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 2, 1])

    with mid:
        if st.button(
            t("cta_button", lang),
            type="primary",
            use_container_width=True,
        ):
            go_to("analyze")
            st.rerun()


# ============================================================================
# SCREEN 2 -- Upload & Analyze
# ============================================================================

def run_analysis(audio_bytes: bytes, lang: str):
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


def render_result_card(result, lang: str) -> None:
    verdict = result.verdict

    confidence_pct = f"{result.confidence * 100:.1f}%"

    lang_eer = (
        f"{result.language_eer * 100:.2f}%"
        if result.language_eer is not None
        else t("lang_eer_unavailable", lang)
    )

    meta = (
        f"{t('meta_confidence', lang)}: {confidence_pct} &nbsp;•&nbsp; "
        f"{t('meta_detected_language', lang)}: "
        f"{result.detected_language} &nbsp;•&nbsp; "
        f"{t('meta_language_eer', lang)}: {lang_eer}"
    )

    if verdict == "real":
        html = ui_kit.alert_card(
            "success",
            f"✅ {t('verdict_real_title', lang)}",
            f"<p>{t('verdict_real_body', lang)}</p>",
            meta,
        )

    elif verdict == "fake":
        html = ui_kit.alert_card(
            "amber",
            f"\U0001f6a8 {t('verdict_fake_title', lang)}",
            f"<p>{t('verdict_fake_body', lang)}</p>",
            meta,
        )

    elif verdict == "uncertain":
        html = ui_kit.alert_card(
            "amber",
            f"⚠️ {t('verdict_uncertain_title', lang)}",
            f"<p>{t('verdict_uncertain_body', lang)}</p>",
            meta,
        )

    else:
        html = ui_kit.alert_card(
            "amber",
            f"⚠️ {t('verdict_unknown_title', lang)}",
            f"<p>{verdict}</p>",
            meta,
        )

    st.markdown(html, unsafe_allow_html=True)

    with st.expander(t("technical_details", lang)):
        st.write(
            f"**Raw score (uncalibrated logit):** "
            f"{result.raw_score:.4f}"
        )
        st.write(
            f"**Audio hash (sha256):** `{result.audio_hash}`"
        )
        st.json(result.metadata)


def render_analyze() -> None:
    lang = current_lang()

    render_header(show_back=True, show_next=False)

    st.markdown(
        f"<h2>{t('screen2_heading', lang)}</h2>",
        unsafe_allow_html=True,
    )

    st.caption(t("language_caption", lang))

    uploaded = st.file_uploader(
        t("upload_label", lang),
        type=ACCEPTED_TYPES,
        label_visibility="collapsed",
    )

    submit = st.button(
        t("analyze_button", lang),
        type="primary",
        use_container_width=True,
    )

    if submit:

        if uploaded is None:
            st.markdown(
                ui_kit.alert_card(
                    "amber",
                    f"⚠️ {t('no_file_title', lang)}",
                    f"<p>{t('no_file_body', lang)}</p>",
                ),
                unsafe_allow_html=True,
            )

        else:
            loader_slot = st.empty()

            with loader_slot:
                components.html(
                    ui_kit.hamster_loader_html(),
                    height=210,
                    scrolling=False,
                )

            try:
                audio_bytes = uploaded.getvalue()

                st.session_state.audio_filename = uploaded.name

                result = run_analysis(
                    audio_bytes,
                    lang,
                )

                st.session_state.result = result
                st.session_state.result_error = None

            except Exception as exc:
                st.session_state.result = None
                st.session_state.result_error = str(exc)

            loader_slot.empty()

    if st.session_state.result_error:

        st.markdown(
            ui_kit.alert_card(
                "amber",
                f"⚠️ {t('error_title', lang)}",
                f"<p>{st.session_state.result_error}</p>",
            ),
            unsafe_allow_html=True,
        )

    elif st.session_state.result is not None:

        render_result_card(
            st.session_state.result,
            lang,
        )

    # =========================================================================
    # PDF EVIDENCE REPORT
    # =========================================================================

    st.markdown(
        f"<h3 style='margin-top:2rem;'>"
        f"{t('pdf_heading', lang)}</h3>",
        unsafe_allow_html=True,
    )

    if st.session_state.result is not None:

        st.markdown(
            '<div class="content-card">'
            '<p style="margin:0;">'
            'Generate a PDF evidence report for this analysis.'
            '</p></div>',
            unsafe_allow_html=True,
        )

        try:

            report_pdf = evidence_report.generate_evidence_report(
                st.session_state.result,
                st.session_state.get(
                    "audio_filename",
                    "uploaded_audio",
                ),
            )

            st.download_button(
                label="📄 Generate Evidence Report",
                data=report_pdf,
                file_name="voice_detection_evidence_report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )

        except Exception as exc:

            st.error(
                f"Could not generate the Evidence Report: {exc}"
            )

    else:

        st.info(
            "Analyze an audio file first to generate the Evidence Report."
        )

    # Pushes the sections below the fold so "appears on scroll" is real.
    st.markdown(
        "<div style='height:12vh;'></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<h3>{t('whatsapp_section_heading', lang)}</h3>",
        unsafe_allow_html=True,
    )

    components.html(
        ui_kit.scroll_reveal_component(
            t("whatsapp_title", lang),
            f"<p>{t('whatsapp_body', lang)}</p>",
            badge_text=t("coming_soon_badge", lang),
            icon_svg=ui_kit.whatsapp_icon_svg(),
        ),
        height=180,
        scrolling=False,
    )

    st.markdown(
        "<div style='height:6vh;'></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<h3>{t('live_call_section_heading', lang)}</h3>",
        unsafe_allow_html=True,
    )

    components.html(
        ui_kit.live_call_companion_html(),
        height=220,
        scrolling=False,
    )

    components.html(
        ui_kit.scroll_reveal_component(
            t("live_call_title", lang),
            f"<p>{t('live_call_body', lang)}</p>",
            badge_text=t("coming_soon_badge", lang),
        ),
        height=170,
        scrolling=False,
    )


# ============================================================================

if st.session_state.screen == "landing":
    render_landing()
else:
    render_analyze()
