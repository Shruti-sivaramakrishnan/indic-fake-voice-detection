"""Append verified completion updates to the two Voice Guard handoff documents."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
LONG_DOC = ROOT / "docs" / "Voice_Guard_Long_Call_Feature_Handoff.docx"
EMOTION_DOC = ROOT / "docs" / "Voice_Guard_Emotion_Robustness_Baseline.docx"


def shade(cell, fill: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    shading = props.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        props.append(shading)
    shading.set(qn("w:fill"), fill)


def set_borders(table) -> None:
    props = table._tbl.tblPr
    borders = props.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        props.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), "D9D9D9")
        borders.append(element)


def add_status_table(document, rows) -> None:
    table = document.add_table(rows=1, cols=3)
    table.autofit = False
    widths = (2.05, 1.05, 3.35)
    headings = ("Item", "Status", "Evidence and meaning")
    for index, (cell, heading, width) in enumerate(
        zip(table.rows[0].cells, headings, widths)
    ):
        cell.width = Pt(width * 72)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        shade(cell, "1F4E78")
        paragraph = cell.paragraphs[0]
        run = paragraph.add_run(heading)
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(10)
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        if row_index % 2:
            for cell in cells:
                shade(cell, "EAF2F8")
        for cell, value, width in zip(cells, row, widths):
            cell.width = Pt(width * 72)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(value)
            run.font.size = Pt(9.5)
    set_borders(table)


def add_page_break(document) -> None:
    paragraph = document.add_paragraph()
    paragraph.add_run().add_break(WD_BREAK.PAGE)


def update_long_call() -> None:
    document = Document(LONG_DOC)
    add_page_break(document)
    document.add_heading("Verified completion update", level=1)
    document.add_paragraph(
        "This update records the end-to-end evidence produced after the original handoff. "
        "The implementation is ready to demonstrate, but the multilingual localization target "
        "has not yet passed and must not be described as fully validated."
    )
    add_status_table(
        document,
        [
            ("Automated tests", "PASS", "48 tests pass, including call analysis, codec robustness, reports, input policy, emotion analysis, acceptance helpers and transcript cues."),
            ("Fifteen minute stress run", "PASS", "A 15-minute synthetic recording completed across 300 windows in 75.62 seconds with about 2.10 GiB maximum resident memory."),
            ("All-real controls", "PASS", "Five controlled five-minute calls across Assamese, Bengali and Bodo produced zero flagged windows."),
            ("Spliced-call localization", "PARTIAL", "Two of five calls localized the 120 to 180 second fake interval within plus or minus six seconds. Three calls missed the boundary target."),
            ("Presentation readiness", "READY", "Upload, progress, timeline, flagged ranges, CSV, PDF and limitation wording can be demonstrated."),
            ("Final accuracy claim", "NOT READY", "The historical v1 split has known group overlap and the 2 of 5 localization result is below acceptance."),
        ],
    )

    document.add_heading("Controlled acceptance result", level=1)
    document.add_paragraph(
        "Five calls were built from frozen-test examples in three languages. Each spliced call "
        "contained 120 seconds of real speech, 60 seconds of synthetic speech and another 120 "
        "seconds of real speech. Five matching all-real controls were also analysed. The run was "
        "for feature acceptance only and did not change thresholds or model weights."
    )
    add_status_table(
        document,
        [
            ("Assamese call 1", "FAIL", "Best flagged range 139.90 to 144.43 seconds."),
            ("Assamese call 2", "FAIL", "Best flagged range 131.76 to 136.29 seconds."),
            ("Bengali call 1", "PASS", "Best flagged range 118.39 to 181.39 seconds."),
            ("Bengali call 2", "PASS", "Best flagged range 119.44 to 180.02 seconds."),
            ("Bodo call 1", "FAIL", "Best flagged range 138.99 to 147.40 seconds."),
        ],
    )
    document.add_paragraph(
        "The correct presentation statement is that long-call analysis operates end to end and "
        "produced no false flags on these five all-real controls, while multilingual fake-section "
        "localization remains a measured weakness."
    )

    document.add_heading("Optional transcript cue analysis", level=1)
    document.add_paragraph(
        "Call Analysis now accepts an optional UTF-8 timestamped transcript CSV with "
        "start_seconds, end_seconds and text columns. It reports lexical sentiment and highlights "
        "configured urgency, secrecy, payment, OTP or credential, authority or familiarity, and "
        "pressure or threat phrases in English, Hindi, Tamil and Kannada. Results retain the "
        "transcript timestamps so reviewers can compare cue locations with the synthetic-voice timeline."
    )
    document.add_paragraph(
        "This is a Phase 2 foundation, not full audio sentiment analysis. Voice Guard does not yet "
        "transcribe the recording, separate speakers, infer intent or prove fraud. A human or ASR "
        "system must provide the timestamped transcript."
    )

    document.add_heading("What to show tomorrow", level=1)
    for text in (
        "Show the existing Single Clip result and codec robustness comparison first.",
        "Switch to Call Analysis and demonstrate progress, timeline, flagged ranges, CSV and PDF export.",
        "Upload a prepared transcript CSV and show cue categories beside their timestamps.",
        "State the measured result exactly: five of five all-real controls stayed clean, while two of five spliced calls met the boundary target.",
        "Close by separating the completed product workflow from the remaining model-validation work.",
    ):
        document.add_paragraph(text, style="List Number")

    document.add_heading("Remaining work and ownership", level=1)
    add_status_table(
        document,
        [
            ("Prompt-disjoint v2 split", "BLOCKED EXTERNALLY", "Exporter and split builder are implemented. Hugging Face repeatedly timed out or returned 503 during the full 31,102-row metadata export."),
            ("Speaker and generator disjointness", "DATA MISSING", "The upstream dataset does not expose stable speaker, source, session or generator lineage."),
            ("V2 training and locked evaluation", "PENDING", "Run the prepared Kaggle training notebook after a trustworthy split is frozen; evaluate once without tuning on the locked test set."),
            ("Automatic transcription", "PENDING", "Add multilingual timestamped ASR and speaker diarization before calling the transcript feature end to end from audio."),
            ("Production load and privacy review", "PENDING", "Containerize, run concurrent 15-minute calls, define retention and monitoring, and complete deployment review."),
            ("Merge and release", "MAINTAINER ACTION", "The feature branch is pushed. Repository maintainers must approve and merge it before release."),
        ],
    )
    document.save(LONG_DOC)


def update_emotion() -> None:
    document = Document(EMOTION_DOC)
    add_page_break(document)
    document.add_heading("Conversation analysis foundation update", level=1)
    document.add_paragraph(
        "The roadmap item for conversation cues has now started as a tested transcript-based "
        "foundation. It complements the completed per-emotion EER diagnostic but answers a "
        "different question: emotion analysis checks whether model error changes across emotional "
        "speech, while conversation analysis looks for review-worthy language in a transcript."
    )
    add_status_table(
        document,
        [
            ("Per-emotion EER", "COMPLETE", "Historical v1 diagnostic for Anger, Disgust, Fear, Happy, Sad and Surprise."),
            ("Transcript CSV validation", "COMPLETE", "UTF-8 schema, timestamp, duration, size, ordering and segment limits are enforced."),
            ("Lexical sentiment", "COMPLETE FOUNDATION", "Positive, negative, mixed or neutral-or-unknown output based on explicit multilingual phrases."),
            ("Manipulation cue categories", "COMPLETE FOUNDATION", "Urgency, secrecy, money, OTP or credentials, authority or familiarity, and pressure or threats."),
            ("Automatic speech recognition", "PENDING", "Audio is not transcribed automatically."),
            ("Intent or fraud classification", "OUT OF CLAIM", "Cue hits guide review and cannot establish intent, identity or fraud."),
        ],
    )
    document.add_heading("Demonstration input", level=1)
    document.add_paragraph(
        "Prepare a small CSV with the columns start_seconds, end_seconds and text. Include one "
        "normal segment and one or two segments containing an urgency plus OTP or payment phrase. "
        "Upload it after the long-call result appears. The interface will show the sentiment label, "
        "category counts and timestamped matched phrases."
    )
    document.add_heading("Next implementation step", level=1)
    document.add_paragraph(
        "Add a locked multilingual ASR component with word or segment timestamps, followed by "
        "speaker diarization and an evaluation set containing both benign and manipulative calls. "
        "Measure cue precision and recall per language before using cue scores for alerts. Keep the "
        "synthetic-voice score and conversation-cue evidence separate in the interface and reports."
    )
    document.save(EMOTION_DOC)


if __name__ == "__main__":
    update_long_call()
    update_emotion()
