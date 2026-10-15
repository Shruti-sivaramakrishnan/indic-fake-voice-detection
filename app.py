"""Voice Guard Streamlit application.

Every analysis is bound to the SHA-256 hash of the selected upload. Replacing
the file clears the previous result and report before the page is rendered.
"""

from __future__ import annotations

import hashlib
import html
import logging
from pathlib import Path

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import streamlit as st

import codec_robustness
import evidence_report
import i18n
import ui_kit
from i18n import t


LOGGER = logging.getLogger("voice_guard.app")
MODEL_IMPORT_ERROR: Exception | None = None

try:
    import model_pipeline as mp
except Exception as exc:  # noqa: BLE001 - replaced with a safe public message
    mp = None
    MODEL_IMPORT_ERROR = exc
    LOGGER.exception("Model pipeline import failed")


st.set_page_config(
    page_title="Voice Guard · AI voice analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(f"<style>{ui_kit.BASE_STYLES}</style>", unsafe_allow_html=True)

for key, default in {
    "ui_language": "English",
    "selected_audio_hash": None,
    "result": None,
    "result_hash": None,
    "result_error": None,
    "audio_filename": None,
    "robustness_report": None,
    "robustness_hash": None,
    "robustness_error": None,
}.items():
    st.session_state.setdefault(key, default)


ACCEPTED_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]


def current_lang() -> str:
    return st.session_state.get("ui_language", "English")


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def clear_analysis_result() -> None:
    st.session_state.result = None
    st.session_state.result_hash = None
    st.session_state.result_error = None
    st.session_state.audio_filename = None
    st.session_state.robustness_report = None
    st.session_state.robustness_hash = None
    st.session_state.robustness_error = None


def sync_selected_upload(audio_bytes: bytes | None) -> str | None:
    selected_hash = hashlib.sha256(audio_bytes).hexdigest() if audio_bytes else None
    if selected_hash != st.session_state.selected_audio_hash:
        st.session_state.selected_audio_hash = selected_hash
        clear_analysis_result()
    return selected_hash


def run_analysis(audio_bytes: bytes):
    if MODEL_IMPORT_ERROR is not None or mp is None:
        raise RuntimeError("model_unavailable")
    return mp.analyze_bytes(audio_bytes)


def run_robustness_analysis(audio_bytes: bytes, original_result):
    if MODEL_IMPORT_ERROR is not None or mp is None:
        raise RuntimeError("model_unavailable")
    return codec_robustness.analyze_codec_robustness(
        audio_bytes,
        mp.analyze_bytes,
        original_result=original_result,
    )


def safe_error_message(exc: Exception, lang: str) -> str:
    if isinstance(exc, codec_robustness.CodecRobustnessError):
        return str(exc)
    if mp is not None and isinstance(exc, mp.AudioValidationError):
        return str(exc)
    if isinstance(exc, ValueError) and str(exc).startswith("Unsupported language hint"):
        return str(exc)
    LOGGER.exception("Audio analysis failed", exc_info=exc)
    return t("analysis_failed_body", lang)


def render_header(lang: str) -> None:
    brand_col, language_col = st.columns([5.3, 1.15], gap="medium")
    with brand_col:
        st.markdown(
            f"""
            <div class="vg-topbar">
              <div>
                <div class="vg-brand">
                  <span class="vg-brand-mark" aria-hidden="true">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                      <path d="M12 3 19 6v5c0 4.6-2.7 8.2-7 10-4.3-1.8-7-5.4-7-10V6l7-3Z"/>
                      <path d="M8.3 12h1.4l1-2.3 2.1 5 1.1-2.7h1.8"/>
                    </svg>
                  </span>
                  <span>{html.escape(t('app_title', lang))}</span>
                </div>
                <div class="vg-status"><span class="vg-status-dot"></span>{html.escape(t('nav_status', lang))}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with language_col:
        st.selectbox(
            t("interface_language_label", lang),
            i18n.LANGUAGES,
            key="ui_language",
            label_visibility="collapsed",
        )


def render_hero(lang: str) -> None:
    chips = "".join(
        f'<span class="vg-chip">{html.escape(label)}</span>'
        for label in (
            t("chip_formats", lang),
            t("chip_languages", lang),
            t("chip_cautious", lang),
        )
    )
    st.markdown(
        f"""
        <section class="vg-hero">
          <div class="vg-eyebrow">{html.escape(t('hero_eyebrow', lang))}</div>
          <h1>{html.escape(t('hero_title_main', lang))} <span>{html.escape(t('hero_title_accent', lang))}</span></h1>
          <p class="vg-hero-copy">{html.escape(t('hero_body', lang))}</p>
          <div class="vg-chips">{chips}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_process_card(lang: str) -> None:
    steps = (
        ("01", t("step_one_title", lang), t("step_one_body", lang)),
        ("02", t("step_two_title", lang), t("step_two_body", lang)),
        ("03", t("step_three_title", lang), t("step_three_body", lang)),
    )
    steps_html = "".join(
        (
            '<div class="vg-step">'
            f'<span class="vg-step-index">{index}</span>'
            f'<div><strong>{html.escape(title)}</strong><p>{html.escape(body)}</p></div>'
            "</div>"
        )
        for index, title, body in steps
    )
    st.markdown(
        f"""
        <aside class="vg-process-card">
          <h3>{html.escape(t('process_title', lang))}</h3>
          <p class="vg-process-intro">{html.escape(t('process_intro', lang))}</p>
          {steps_html}
          <div class="vg-privacy-strip">◈&nbsp; {html.escape(t('session_note', lang))}</div>
        </aside>
        """,
        unsafe_allow_html=True,
    )


def render_result(result, lang: str) -> None:
    p_fake = min(1.0, max(0.0, float(result.metadata.get("p_fake_calibrated", 0.5))))
    score_value = p_fake * 100
    score_pct = f"{score_value:.1f}%"
    raw_duration = result.metadata.get("duration_seconds")
    duration = f"{float(raw_duration):.1f}" if raw_duration is not None else "—"

    if result.verdict == "real":
        css_kind = "real"
        title_key, body_key = "verdict_real_title", "verdict_real_body"
    elif result.verdict == "fake":
        css_kind = "fake"
        title_key, body_key = "verdict_fake_title", "verdict_fake_body"
    else:
        css_kind = "uncertain"
        title_key, body_key = "verdict_uncertain_title", "verdict_uncertain_body"

    st.markdown(
        f"""
        <section class="vg-result-shell {css_kind}">
          <div>
            <div class="vg-result-label">{html.escape(t('result_label', lang))}</div>
            <h2>{html.escape(t(title_key, lang))}</h2>
            <p>{html.escape(t(body_key, lang))}</p>
            <p class="vg-score-note">{html.escape(t('score_explanation', lang))}</p>
          </div>
          <div class="vg-score-ring" style="--score:{score_value:.3f}">
            <div class="vg-score-ring-inner">
              <strong>{score_pct}</strong>
              <span>{html.escape(t('score_ring_label', lang))}</span>
            </div>
          </div>
        </section>
        <div class="vg-scale-card">
          <div class="vg-scale-track">
            <span class="vg-scale-marker" style="--score:{score_value:.3f}"></span>
          </div>
          <div class="vg-scale-labels">
            <span>{html.escape(t('likely_genuine_end', lang))}</span>
            <span>{html.escape(t('likely_synthetic_end', lang))}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    guidance_col, transparency_col = st.columns(2, gap="medium")
    with guidance_col:
        st.markdown(
            ui_kit.alert_card(
                "warning",
                html.escape(t("safety_action_title", lang)),
                f"<p>{html.escape(t('safety_action_body', lang))}</p>",
            ),
            unsafe_allow_html=True,
        )
    with transparency_col:
        st.markdown(
            f"""
            <div class="vg-info-card">
              <div class="vg-info-icon">◎</div>
              <h4>{html.escape(t('transparency_title', lang))}</h4>
              <p>{html.escape(t('transparency_body', lang))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(t("permanent_disclaimer", lang))
    with st.expander(t("technical_details", lang)):
        details_left, details_right = st.columns(2)
        with details_left:
            st.write(f"**{t('ai_likelihood_score', lang)}:** {score_pct}")
            st.write(f"**{t('duration_label', lang)}:** {duration}s")
            st.write(f"**{t('raw_score_label', lang)}:** {result.raw_score:.4f}")
        with details_right:
            st.write(
                f"**{t('sample_rate_label', lang)}:** "
                f"{result.metadata.get('sample_rate_in', '—')} Hz"
            )
            st.write(
                f"**{t('policy_label', lang)}:** "
                f"`{result.metadata.get('decision_policy_id', 'unknown')}`"
            )
            st.write(f"**SHA-256:** `{result.audio_hash}`")
        quality = result.metadata.get("audio_quality", {})
        if quality.get("quality_warning"):
            st.warning(t("quality_warning_clipping", lang))


def render_robustness_report(report, lang: str) -> None:
    """Render a compact original-versus-channel comparison."""
    rows = []
    for condition in report.conditions:
        if condition.succeeded:
            delta = condition.delta_p_fake or 0.0
            rows.append(
                {
                    t("robustness_condition", lang): condition.label,
                    t("robustness_score", lang): f"{condition.p_fake * 100:.1f}%",
                    t("robustness_change", lang): (
                        "—" if condition.condition_id == "original"
                        else f"{delta * 100:+.1f} pp"
                    ),
                    t("robustness_verdict", lang): condition.verdict.title(),
                    t("robustness_stability", lang): (
                        t("robustness_changed", lang)
                        if condition.verdict_changed
                        else t("robustness_unchanged", lang)
                    ),
                }
            )
        else:
            rows.append(
                {
                    t("robustness_condition", lang): condition.label,
                    t("robustness_score", lang): "—",
                    t("robustness_change", lang): "—",
                    t("robustness_verdict", lang): t("robustness_failed", lang),
                    t("robustness_stability", lang): "—",
                }
            )

    panel = st.container(border=True)
    with panel:
        st.markdown(f"### {t('robustness_title', lang)}")
        st.caption(t("robustness_body", lang))
        st.dataframe(rows, hide_index=True, use_container_width=True)

        if report.transformed_success_count == 0:
            st.warning(t("robustness_no_comparisons", lang))
        elif report.verdict_change_count:
            st.warning(
                t("robustness_summary_changed", lang).format(
                    count=report.verdict_change_count,
                    total=report.transformed_success_count,
                )
            )
        else:
            st.success(t("robustness_summary_stable", lang))

        st.caption(
            t("robustness_max_change", lang).format(
                value=report.max_absolute_delta * 100
            )
        )
        if report.failed_conditions:
            failed = ", ".join(condition.label for condition in report.failed_conditions)
            st.warning(t("robustness_partial_failure", lang).format(conditions=failed))
        st.caption(t("robustness_limit", lang))


def render_report_download(result, filename: str, lang: str) -> None:
    try:
        report_pdf = evidence_report.generate_analysis_report(result, filename)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("PDF report generation failed", exc_info=exc)
        st.warning(t("pdf_error", lang))
        return

    safe_stem = "".join(
        char for char in Path(filename).stem if char.isalnum() or char in "-_"
    )[:60] or "audio"
    report_panel = st.container(border=True)
    with report_panel:
        st.markdown(
            f'<div class="vg-section-kicker">{html.escape(t("pdf_heading", lang))}</div>',
            unsafe_allow_html=True,
        )
        st.caption(t("pdf_body", lang))
        st.download_button(
            label=t("pdf_button", lang),
            data=report_pdf,
            file_name=f"voice_guard_{safe_stem}_analysis.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def render_app() -> None:
    lang = current_lang()
    render_header(lang)
    render_hero(lang)

    upload_col, process_col = st.columns([1.55, 0.85], gap="large")
    with upload_col:
        upload_panel = st.container(border=True)
        with upload_panel:
            st.markdown(
                f"""
                <div class="vg-section-kicker">{html.escape(t('workspace_kicker', lang))}</div>
                <h2 class="vg-section-title">{html.escape(t('how_it_works_title', lang))}</h2>
                <p class="vg-section-copy">{html.escape(t('how_it_works_body', lang))}</p>
                """,
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                t("upload_label", lang),
                type=ACCEPTED_TYPES,
                help=t("upload_help", lang),
                label_visibility="collapsed",
                key="audio_upload",
            )
            audio_bytes = uploaded.getvalue() if uploaded is not None else None
            selected_hash = sync_selected_upload(audio_bytes)

            if uploaded is not None and audio_bytes is not None:
                st.audio(audio_bytes)
                st.markdown(
                    f"""
                    <div class="vg-file-meta">
                      <span class="vg-file-name">{html.escape(uploaded.name)}</span>
                      <span class="vg-file-size">{format_bytes(len(audio_bytes))}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            analyze = st.button(
                t("analyze_button", lang),
                type="primary",
                use_container_width=True,
                disabled=uploaded is None,
            )

            if analyze and audio_bytes is not None:
                st.session_state.result_error = None
                st.session_state.robustness_report = None
                st.session_state.robustness_hash = None
                st.session_state.robustness_error = None
                with st.spinner(t("analyzing_status", lang)):
                    try:
                        result = run_analysis(audio_bytes)
                        st.session_state.result = result
                        st.session_state.result_hash = selected_hash
                        st.session_state.audio_filename = uploaded.name
                    except Exception as exc:  # noqa: BLE001
                        clear_analysis_result()
                        st.session_state.result_error = safe_error_message(exc, lang)

            if st.session_state.result_error:
                st.markdown(
                    ui_kit.alert_card(
                        "danger",
                        html.escape(t("error_title", lang)),
                        f"<p>{html.escape(st.session_state.result_error)}</p>",
                    ),
                    unsafe_allow_html=True,
                )

    with process_col:
        render_process_card(lang)

    result_is_current = (
        st.session_state.result is not None
        and selected_hash is not None
        and st.session_state.result_hash == selected_hash
    )
    if result_is_current:
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        render_result(st.session_state.result, lang)

        run_robustness = st.button(
            t("robustness_button", lang),
            use_container_width=True,
            key="run_robustness",
        )
        if run_robustness and audio_bytes is not None:
            st.session_state.robustness_error = None
            with st.spinner(t("robustness_running", lang)):
                try:
                    report = run_robustness_analysis(
                        audio_bytes,
                        st.session_state.result,
                    )
                    st.session_state.robustness_report = report
                    st.session_state.robustness_hash = selected_hash
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Robustness analysis failed", exc_info=exc)
                    st.session_state.robustness_report = None
                    st.session_state.robustness_hash = None
                    st.session_state.robustness_error = safe_error_message(exc, lang)

        robustness_is_current = (
            st.session_state.robustness_report is not None
            and st.session_state.robustness_hash == selected_hash
        )
        if st.session_state.robustness_error:
            st.warning(st.session_state.robustness_error)
        if robustness_is_current:
            render_robustness_report(st.session_state.robustness_report, lang)

        render_report_download(
            st.session_state.result,
            st.session_state.audio_filename or "uploaded_audio",
            lang,
        )

    st.markdown(
        f"""
        <footer class="vg-footer">
          <strong>{html.escape(t('footer_product', lang))}</strong>
          <span>{html.escape(t('privacy_note', lang))}</span>
        </footer>
        """,
        unsafe_allow_html=True,
    )


render_app()
