"""Generate the user-downloadable Voice Guard automated analysis report."""

from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


INK = colors.HexColor("#1A1A1A")
YELLOW = colors.HexColor("#FFD93D")
CREAM = colors.HexColor("#FFFDF5")
MUTED = colors.HexColor("#6B6350")
BORDER = colors.HexColor("#CBD5E1")
AMBER_BG = colors.HexColor("#FFF3D6")
AMBER_TEXT = colors.HexColor("#5A3400")


def _safe(value: Any) -> str:
    return escape(str(value), quote=True)


def _hash_for_wrapping(value: str) -> str:
    safe = _safe(value)
    return " ".join(safe[index : index + 16] for index in range(0, len(safe), 16))


def _footer(canvas: Canvas, doc: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setStrokeColor(YELLOW)
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(18 * mm, 8.5 * mm, "Voice Guard - automated model analysis")
    canvas.drawRightString(
        A4[0] - 18 * mm,
        8.5 * mm,
        f"Page {canvas.getPageNumber()}",
    )
    canvas.restoreState()


def generate_analysis_report(result: Any, filename: str = "uploaded_audio") -> bytes:
    """Return a polished PDF record without representing the result as proof."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=14 * mm,
        bottomMargin=19 * mm,
        title="Voice Guard Automated Analysis Report",
        author="Voice Guard",
        subject="Automated AI voice model output and limitations",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=21,
        textColor=INK,
        spaceAfter=5,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        leading=12,
        textColor=MUTED,
        spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=INK,
        spaceBefore=6,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.3,
        leading=13.5,
        textColor=INK,
    )
    small_style = ParagraphStyle(
        "Small",
        parent=body_style,
        fontSize=7.8,
        leading=9.6,
        wordWrap="CJK",
    )
    warning_style = ParagraphStyle(
        "Warning",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=AMBER_TEXT,
    )

    def cell(value: Any, style: ParagraphStyle = small_style) -> Paragraph:
        return Paragraph(_safe(value), style)

    def key_value_table(rows: list[tuple[Any, Any]], widths=(52 * mm, 115 * mm)) -> Table:
        data = [[cell(key), cell(value)] for key, value in rows]
        table = Table(data, colWidths=list(widths), hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), CREAM),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ]
            )
        )
        return table

    metadata = result.metadata or {}
    p_fake = float(metadata.get("p_fake_calibrated", 0.5))
    verdict = str(result.verdict).lower()
    verdict_label = {
        "real": "LIKELY GENUINE",
        "fake": "LIKELY SYNTHETIC",
        "uncertain": "INCONCLUSIVE",
    }.get(verdict, "UNRECOGNIZED")
    interpretation = {
        "real": "The clip is more consistent with genuine speech in this model's learned patterns.",
        "fake": "The clip is more consistent with synthetic speech in this model's learned patterns.",
        "uncertain": "The model abstained because its score falls inside the inconclusive band.",
    }.get(verdict, "The model did not return a recognized result.")

    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    story: list[Any] = [
        Paragraph("Voice Guard<br/>Automated Analysis Report", title_style),
        Paragraph(
            "A record of one model run. This document is not forensic evidence.",
            subtitle_style,
        ),
        HRFlowable(width="100%", thickness=1.2, color=YELLOW),
        Spacer(1, 7),
        Paragraph("Analysis summary", heading_style),
    ]

    summary_table = Table(
        [
            [Paragraph("RESULT", small_style), Paragraph("AI-LIKELIHOOD MODEL SCORE", small_style)],
            [Paragraph(verdict_label, warning_style), Paragraph(f"{p_fake * 100:.1f}%", warning_style)],
        ],
        colWidths=[83.5 * mm, 83.5 * mm],
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), AMBER_BG),
                ("BOX", (0, 0), (-1, -1), 1, YELLOW),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, YELLOW),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend(
        [
            summary_table,
            Spacer(1, 4),
            Paragraph(interpretation, body_style),
            Paragraph("Input record", heading_style),
            key_value_table(
                [
                    ("Generated", generated_at),
                    ("Audio file", filename),
                    ("Duration", f"{metadata.get('duration_seconds', 'N/A')} seconds"),
                    ("Input sample rate", f"{metadata.get('sample_rate_in', 'N/A')} Hz"),
                    ("SHA-256", _hash_for_wrapping(result.audio_hash or "N/A")),
                ]
            ),
            Paragraph("Model record", heading_style),
            key_value_table(
                [
                    ("Frontend", metadata.get("frontend", "N/A")),
                    ("Checkpoint", metadata.get("checkpoint", "N/A")),
                    ("Checkpoint epoch", metadata.get("checkpoint_epoch", "N/A")),
                    ("Temperature scaling", metadata.get("temperature", "N/A")),
                    ("Decision policy", metadata.get("decision_policy_id", "N/A")),
                    ("Raw fake logit", f"{result.raw_score:.6f}"),
                ]
            ),
        ]
    )

    quality = metadata.get("audio_quality", {})
    if quality:
        story.extend(
            [
                Paragraph("Input quality checks", heading_style),
                key_value_table(
                    [
                        ("Estimated active speech", f"{quality.get('active_speech_seconds_estimate', 'N/A')} seconds"),
                        ("Signal RMS", f"{float(quality.get('rms', 0.0)):.6f}"),
                        ("Peak amplitude", f"{float(quality.get('peak', 0.0)):.6f}"),
                        ("Quality warning", quality.get("quality_warning") or "None detected"),
                    ]
                ),
            ]
        )

    story.append(
        KeepTogether(
            [
                Paragraph("Safety guidance and limitations", heading_style),
                Table(
                    [
                        [
                            Paragraph(
                                "If this recording asks for money, a code, or urgent help, contact the person using a number you already know before acting.<br/><br/>"
                                "Automated analysis can be wrong, especially with noise, compression, unseen generators, very short speech, or playback recordings. The score is a temperature-scaled model output, not a guaranteed real-world probability. This report is not proof of authenticity, identity, intent, or deception.",
                                warning_style,
                            )
                        ]
                    ],
                    colWidths=[167 * mm],
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), AMBER_BG),
                            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#B85C00")),
                            ("LEFTPADDING", (0, 0), (-1, -1), 9),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                            ("TOPPADDING", (0, 0), (-1, -1), 8),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ]
                    ),
                ),
            ]
        )
    )

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()


def generate_evidence_report(result: Any, filename: str = "uploaded_audio") -> bytes:
    """Backward-compatible alias; new code should use generate_analysis_report."""
    return generate_analysis_report(result, filename)


def generate_call_analysis_report(report: Any, filename: str = "call_recording") -> bytes:
    """Create a PDF report for a full phone-call analysis."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=14 * mm,
        bottomMargin=19 * mm,
        title="Voice Guard Call Analysis Report",
        author="Voice Guard",
        subject="Automated call-level AI voice analysis",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CallReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=21,
        textColor=INK,
        spaceAfter=5,
    )
    subtitle_style = ParagraphStyle(
        "CallReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        leading=12,
        textColor=MUTED,
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "CallReportHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=INK,
        spaceBefore=9,
        spaceAfter=5,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "CallReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        textColor=INK,
    )
    small_style = ParagraphStyle(
        "CallReportSmall",
        parent=body_style,
        fontSize=7.5,
        leading=9.5,
        wordWrap="CJK",
    )
    warning_style = ParagraphStyle(
        "CallReportWarning",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=AMBER_TEXT,
    )

    def seconds(value: float) -> str:
        total = max(0, int(value))
        return f"{total // 60:02d}:{total % 60:02d}"

    def make_cell(value: Any, style=small_style) -> Paragraph:
        return Paragraph(_safe(value), style)

    story = [
        Paragraph("Voice Guard", title_style),
        Paragraph("Phone Call Analysis Report", subtitle_style),
        HRFlowable(width="100%", thickness=1, color=YELLOW),
        Spacer(1, 8),
        Paragraph("Recording summary", heading_style),
    ]

    summary_rows = [
        ("Audio file", Path(filename).name),
        ("Recording duration", f"{report.duration_seconds:.2f} seconds"),
        ("Estimated speech duration", f"{report.speech_duration_seconds:.2f} seconds"),
        ("Analysis windows", str(report.windows_analysed)),
        ("Flagged windows", str(report.flagged_windows)),
        ("Flagged share", f"{report.flagged_share:.1%}"),
        ("Highest AI likelihood score", f"{report.highest_p_fake:.1%}"),
        ("Timestamp of highest score", seconds(report.highest_score_timestamp)),
    ]

    summary_table = Table(
        [[make_cell(k), make_cell(v)] for k, v in summary_rows],
        colWidths=[52 * mm, 115 * mm],
        hAlign="LEFT",
    )
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), CREAM),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(summary_table)

    story.append(Paragraph("Window-by-window results", heading_style))
    window_rows = [[
        make_cell("Start"),
        make_cell("End"),
        make_cell("AI likelihood"),
        make_cell("Verdict"),
    ]]

    for score in report.window_scores:
        window_rows.append([
            make_cell(seconds(score.start_seconds)),
            make_cell(seconds(score.end_seconds)),
            make_cell(f"{score.p_fake:.1%}"),
            make_cell(score.verdict.capitalize()),
        ])

    if len(window_rows) == 1:
        window_rows.append([make_cell("No windows analysed"), "", "", ""])

    window_table = Table(
        window_rows,
        colWidths=[35 * mm, 35 * mm, 48 * mm, 49 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    window_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), CREAM),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(window_table)

    story.append(Paragraph("Flagged time ranges", heading_style))
    if report.flagged_ranges:
        range_rows = [[make_cell("Start"), make_cell("End")]]
        for item in report.flagged_ranges:
            range_rows.append([
                make_cell(seconds(item.start_seconds)),
                make_cell(seconds(item.end_seconds)),
            ])
        range_table = Table(
            range_rows,
            colWidths=[83.5 * mm, 83.5 * mm],
            repeatRows=1,
        )
        range_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), CREAM),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(range_table)
    else:
        story.append(Paragraph("No windows were flagged in this analysis.", body_style))

    story.extend([
        Spacer(1, 10),
        Paragraph("Important limitation", heading_style),
        Table(
            [[Paragraph(
                "This report contains automated model estimates, not proof that a "
                "recording is genuine or AI-generated. Scores can be affected by "
                "background noise, compression, short speech, recording quality, "
                "and voice generators not represented in the model's training data. "
                "Review the recording and consider independent verification before "
                "making important decisions.",
                warning_style,
            )]],
            colWidths=[167 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), AMBER_BG),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#B85C00")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]),
        ),
    ])

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
