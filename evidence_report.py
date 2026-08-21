
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


def generate_evidence_report(result, filename="uploaded_audio"):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AI Voice Detection Evidence Report",
        author="Voice Guard",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
    )

    story = []

    story.append(
        Paragraph(
            "AI Voice Detection<br/>Evidence Report",
            title_style,
        )
    )

    generated_at = datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )

    story.append(
        Paragraph(
            "Automated analysis report generated from the live model prediction",
            subtitle_style,
        )
    )

    story.append(HRFlowable(width="100%", thickness=1))
    story.append(Spacer(1, 8))

    # Report information
    story.append(Paragraph("Report Information", heading_style))

    report_data = [
        ["Generated", generated_at],
        ["Audio file", filename],
    ]

    table = Table(
        report_data,
        colWidths=[42 * mm, 125 * mm],
    )

    table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(table)

    # Detection result
    story.append(Paragraph("Detection Result", heading_style))

    confidence = f"{result.confidence * 100:.2f}%"

    detection_data = [
        ["Verdict", str(result.verdict).upper()],
        ["Confidence", confidence],
        ["Raw model score", f"{result.raw_score:.6f}"],
    ]

    table = Table(
        detection_data,
        colWidths=[55 * mm, 112 * mm],
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.append(table)

    # Language
    story.append(Paragraph("Language Information", heading_style))

    detected_language = result.detected_language or "unknown"

    language_eer = (
        f"{result.language_eer * 100:.2f}%"
        if result.language_eer is not None
        else "N/A"
    )

    language_data = [
        ["Detected language", detected_language],
        ["Language EER", language_eer],
    ]

    table = Table(
        language_data,
        colWidths=[55 * mm, 112 * mm],
    )

    table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(table)

    # Audio evidence
    story.append(Paragraph("Audio Evidence", heading_style))

    audio_hash = result.audio_hash or "N/A"
    metadata = result.metadata or {}

    duration = metadata.get("duration_seconds", "N/A")
    sample_rate = metadata.get("sample_rate_in", "N/A")

    audio_data = [
        ["SHA-256", audio_hash],
        ["Duration", f"{duration} seconds"],
        ["Input sample rate", f"{sample_rate} Hz"],
    ]

    table = Table(
        audio_data,
        colWidths=[55 * mm, 112 * mm],
    )

    table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(table)

    # Model information
    story.append(Paragraph("Model Information", heading_style))

    frontend = metadata.get("frontend", "N/A")
    checkpoint = metadata.get("checkpoint", "N/A")
    checkpoint_epoch = metadata.get("checkpoint_epoch", "N/A")
    temperature = metadata.get("temperature", "N/A")
    calibrated = metadata.get("calibrated", False)

    model_data = [
        ["Frontend", frontend],
        ["Checkpoint", checkpoint],
        ["Checkpoint epoch", str(checkpoint_epoch)],
        ["Temperature", str(temperature)],
        ["Calibrated", "Yes" if calibrated else "No"],
    ]

    table = Table(
        model_data,
        colWidths=[55 * mm, 112 * mm],
    )

    table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(table)

    # Interpretation
    story.append(Paragraph("Interpretation", heading_style))

    verdict = str(result.verdict).lower()

    if verdict == "fake":
        interpretation = (
            "The model classified the submitted audio as synthetic "
            "or potentially AI-generated speech."
        )
    elif verdict == "real":
        interpretation = (
            "The model classified the submitted audio as bonafide "
            "or genuine speech."
        )
    else:
        interpretation = (
            "The model could not make a definitive fake/bonafide "
            "classification for this audio."
        )

    story.append(Paragraph(interpretation, body_style))

    # Disclaimer
    story.append(Paragraph("Important Disclaimer", heading_style))

    disclaimer = (
        "This report represents an automated model prediction and "
        "should not be treated as definitive proof of authenticity "
        "or deception. The result should be interpreted together "
        "with the confidence and technical evidence provided in "
        "this report."
    )

    story.append(Paragraph(disclaimer, body_style))

    doc.build(story)

    return buffer.getvalue()
