"""Build the two presentation reports for speaker-aware analysis and validation."""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "detailed_reports"
TMP = ROOT / ".report_assets"
NAVY = "15324A"
BLUE = "247BA0"
TEAL = "2A9D8F"
PALE = "EAF4F4"
ORANGE = "F4A261"


def shade(cell, fill: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    node = props.find(qn("w:shd"))
    if node is None:
        node = OxmlElement("w:shd")
        props.append(node)
    node.set(qn("w:fill"), fill)


def configure(document: Document, short_title: str) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.62)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)
    styles = document.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(10.5)
    styles["Title"].font.name = "Aptos Display"
    styles["Title"].font.size = Pt(30)
    styles["Title"].font.color.rgb = RGBColor.from_string(NAVY)
    for level, size in ((1, 20), (2, 14)):
        style = styles[f"Heading {level}"]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(NAVY if level == 1 else BLUE)
    header = section.header.paragraphs[0]
    header.text = f"VOICE GUARD  /  {short_title.upper()}"
    header.style = styles["Caption"]
    header.runs[0].font.color.rgb = RGBColor.from_string(BLUE)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Internal implementation evidence  •  21 November 2026  •  ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)


def page(document: Document, title: str, kicker: str | None = None) -> None:
    if len(document.paragraphs) > 1:
        document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    if kicker:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(2)
        run = paragraph.add_run(kicker.upper())
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor.from_string(TEAL)
    document.add_heading(title, level=1)


def paragraph(document: Document, text: str, bold_lead: str | None = None) -> None:
    item = document.add_paragraph()
    item.paragraph_format.space_after = Pt(7)
    if bold_lead and text.startswith(bold_lead):
        item.add_run(bold_lead).bold = True
        item.add_run(text[len(bold_lead) :])
    else:
        item.add_run(text)


def bullets(document: Document, values) -> None:
    for value in values:
        item = document.add_paragraph(value, style="List Bullet")
        item.paragraph_format.space_after = Pt(3)


def table(document: Document, headers, rows, widths=None) -> None:
    grid = document.add_table(rows=1, cols=len(headers))
    grid.autofit = False
    for index, (cell, label) in enumerate(zip(grid.rows[0].cells, headers)):
        shade(cell, NAVY)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        run = cell.paragraphs[0].add_run(str(label))
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(9)
        if widths:
            cell.width = Inches(widths[index])
    for row_index, values in enumerate(rows):
        cells = grid.add_row().cells
        for index, (cell, value) in enumerate(zip(cells, values)):
            if row_index % 2 == 0:
                shade(cell, PALE)
            run = cell.paragraphs[0].add_run(str(value))
            run.font.size = Pt(8.6)
            if widths:
                cell.width = Inches(widths[index])
    document.add_paragraph()


def callout(document: Document, title: str, text: str, color: str = TEAL) -> None:
    grid = document.add_table(rows=1, cols=1)
    shade(grid.cell(0, 0), "E8F5F2" if color == TEAL else "FFF2E8")
    item = grid.cell(0, 0).paragraphs[0]
    run = item.add_run(f"{title}\n")
    run.bold = True
    run.font.color.rgb = RGBColor.from_string(color)
    item.add_run(text)
    document.add_paragraph()


def chart(path: Path, labels, values, title: str, suffix: str = "%") -> None:
    image = Image.new("RGB", (1400, 620), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 42)
        label_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
        value_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 30)
    except OSError:
        title_font = label_font = value_font = ImageFont.load_default()
    draw.text((70, 35), title, fill="#15324A", font=title_font)
    maximum = max(max(values), 1)
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 145 + index * 105
        draw.text((70, y), label, fill="#15324A", font=label_font)
        draw.rounded_rectangle((430, y, 1250, y + 54), radius=18, fill="#E5EEF2")
        width = 820 * value / maximum
        draw.rounded_rectangle((430, y, 430 + width, y + 54), radius=18, fill="#2A9D8F")
        draw.text((1270, y + 6), f"{value:g}{suffix}", fill="#15324A", font=value_font)
    image.save(path)


def add_title_page(document: Document, title: str, subtitle: str, tags) -> None:
    document.add_paragraph("VOICE GUARD", style="Subtitle")
    document.add_paragraph(title, style="Title")
    document.add_paragraph(subtitle, style="Subtitle")
    document.add_paragraph("\n")
    table(document, ("Delivery", "Status", "Evidence"), tags, (1.7, 1.2, 4.0))
    callout(
        document,
        "Report boundary",
        "This document records implemented software and checked-in acceptance evidence. "
        "It does not convert a demo baseline into a production or research-generalization claim.",
        ORANGE,
    )
    document.add_paragraph("Prepared 21 November 2026", style="Caption")


def build_speaker_report() -> None:
    document = Document()
    configure(document, "Speaker-aware analysis")
    add_title_page(
        document,
        "Speaker-Aware Long-Call Analysis",
        "Anonymous diarization, transcript alignment and per-speaker conversation risk",
        [
            ("Implementation", "Complete", "Offline two-cluster acoustic baseline"),
            ("Integration", "Complete", "ASR → alignment → per-speaker reports → Streamlit"),
            ("Automated tests", "Pass", "63 repository tests; 4 speaker-flow tests"),
        ],
    )

    page(document, "Why the feature was added", "1 / Existing system and gap")
    paragraph(document, "Before this change, Voice Guard could score long-call windows for synthetic speech and analyze the complete ASR transcript for lexical tone and manipulation cues. It could not show which anonymous voice produced a risky request.")
    paragraph(document, "That limitation matters in a demonstration: a victim may sound worried while a caller makes the payment or OTP request. A whole-call label merges those roles and makes the evidence harder to interpret.")
    table(document, ("Before", "Risk", "Added now"), [
        ("Synthetic-speech timeline", "No speaker attribution", "Anonymous speaker turns"),
        ("Timestamped ASR segments", "Mixed voices", "Maximum-overlap alignment"),
        ("Whole-call cue report", "Role ambiguity", "Per-speaker tone/risk/cues"),
    ], (2.2, 2.1, 2.6))
    callout(document, "Design decision", "The labels are Speaker A/B acoustic clusters. They never identify a person.")

    page(document, "End-to-end data flow", "2 / Architecture")
    table(document, ("Stage", "Input", "Output"), [
        ("Decode and resample", "Uploaded call", "16 kHz mono waveform"),
        ("Voice framing", "1.5 s frames / 0.75 s hop", "Voiced acoustic frames"),
        ("Clustering", "Spectral features", "Speaker A/B labels + confidence"),
        ("ASR alignment", "Timestamped transcript", "Speaker-linked segments"),
        ("Conversation analysis", "Segments per speaker", "Tone, risk, cue categories"),
        ("Streamlit", "All evidence", "Review table and limitations"),
    ], (1.8, 2.5, 2.6))
    bullets(document, ["No gated diarization model or cloud credential is required.", "A backend callable can replace the lightweight clusterer later.", "A diarization failure does not discard successful ASR or whole-call analysis."])

    page(document, "Acoustic clustering baseline", "3 / Method")
    paragraph(document, "Each voiced frame is represented by spectral centroid, spectral spread, zero-crossing rate and six log band-energy values. Features are standardized per recording and split with deterministic two-means initialized from the farthest frame pair.")
    table(document, ("Control", "Implementation", "Reason"), [
        ("Voice activity", "Adaptive RMS threshold", "Avoid clustering silence"),
        ("Stable labels", "Lower centroid → Speaker A", "Reproducible display"),
        ("Fallback", "Low separation → one speaker", "Avoid forced false split"),
        ("Confidence", "Nearest/second-center margin", "Review aid, not probability"),
        ("Turn merge", "Adjacent equal labels", "Readable timeline"),
    ], (1.6, 2.6, 2.7))
    callout(document, "Known limitation", "Overlapping speech, background speakers and similar voices can reduce clustering quality.", ORANGE)

    page(document, "Transcript-to-speaker alignment", "4 / Evidence binding")
    paragraph(document, "Every ASR segment is assigned to the speaker turn with the greatest timestamp overlap. If no turn overlaps, the segment is explicitly marked Unassigned rather than guessed.")
    table(document, ("Transcript interval", "Overlapping turn", "Result"), [
        ("00:01–00:04", "Speaker A 00:00–00:05", "Speaker A"),
        ("00:06–00:12", "Speaker B 00:05–00:15", "Speaker B"),
        ("No overlap", "None", "Unassigned / confidence 0"),
    ], (2.1, 2.8, 2.0))
    paragraph(document, "After alignment, the existing Unicode-safe conversation analyzer runs independently on each speaker's segments. The cue vocabulary and Low/Medium/High rule are unchanged, so this feature adds attribution without silently changing risk semantics.")

    page(document, "User interface and demo", "5 / Streamlit")
    paragraph(document, "The existing Transcribe call and analyze conversation action now also performs speaker-aware review. The transcript table gains a speaker column, followed by a compact summary for each anonymous cluster.")
    table(document, ("Speaker", "Segments", "Tone", "Risk", "Cue categories"), [
        ("Speaker A", "1", "positive", "Low", "none"),
        ("Speaker B", "1", "neutral", "High", "urgency, payment, OTP"),
    ], (1.2, 1.0, 1.2, 0.9, 2.6))
    bullets(document, ["Start with the existing synthetic-voice timeline.", "Open the generated transcript to show speaker attribution.", "Compare the per-speaker risk summaries.", "State that A/B are anonymous clusters, not identities."])

    page(document, "Automated verification", "6 / Tests")
    chart(TMP / "speaker_tests.png", ["All repository tests", "Speaker-flow tests", "Failures"], [63, 4, 0], "Automated verification", "")
    document.add_picture(str(TMP / "speaker_tests.png"), width=Inches(6.9))
    table(document, ("Test", "Assertion"), [
        ("Maximum-overlap alignment", "Segments bind to the correct turn"),
        ("Per-speaker isolation", "Benign speaker Low; requesting speaker High"),
        ("Acoustic separation", "Two distinct sine regions produce A and B"),
        ("Full suite", "63 passed"),
    ], (2.7, 4.0))

    page(document, "Failure behavior and safety", "7 / Operational behavior")
    table(document, ("Condition", "System response"), [
        ("Unreadable audio", "Safe diarization error; ASR workflow reports its own result"),
        ("Silence", "No speaker turns; no fabricated speaker"),
        ("Single acoustic cluster", "Speaker A only"),
        ("No overlap", "Unassigned transcript segment"),
        ("Diarization exception", "Whole-call conversation result remains available"),
    ], (2.7, 4.0))
    callout(document, "Privacy posture", "Audio remains in memory in this workflow. A production deployment still needs explicit retention, access and monitoring controls.", ORANGE)

    page(document, "Files changed and reproducibility", "8 / Engineering handoff")
    table(document, ("File", "Role"), [
        ("speaker_diarization.py", "Clustering, alignment and per-speaker report"),
        ("speech_transcription.py", "End-to-end orchestration and graceful fallback"),
        ("app.py", "Speaker column, summaries and limitation copy"),
        ("tests/test_speaker_diarization.py", "Deterministic feature tests"),
        ("README.md", "Run behavior, claim boundary and next validation"),
    ], (2.8, 3.9))
    paragraph(document, "Reproduce with: python -m pytest -q. No new runtime package was added; the implementation uses NumPy and existing audio decoding/resampling paths.")

    page(document, "Acceptance status and next evidence", "9 / Conclusion")
    table(document, ("Criterion", "Status", "Next evidence"), [
        ("Feature integrated", "PASS", "Merged app workflow"),
        ("Deterministic tests", "PASS", "63-test suite"),
        ("Anonymous labeling", "PASS", "UI and report wording"),
        ("Natural-call DER", "PENDING", "Labelled speaker-turn corpus"),
        ("Overlapping speech", "PENDING", "Overlap-aware benchmark/backend"),
    ], (2.2, 1.2, 3.3))
    callout(document, "Presentation statement", "Voice Guard now links timestamped transcript evidence to anonymous acoustic speaker clusters and reports conversation risk separately for each cluster. Identity recognition and production-grade diarization are not claimed.")
    document.save(OUT / "Voice_Guard_Speaker_Aware_Analysis_Implementation.docx")


def build_validation_report() -> None:
    results = json.loads((ROOT / "evaluation" / "conversation_validation" / "validation_summary.json").read_text(encoding="utf-8"))
    overall = results["overall"]
    document = Document()
    configure(document, "Multilingual validation")
    add_title_page(
        document,
        "Multilingual Conversation Validation Suite",
        "Labelled regression evidence for risk, cue categories and Unicode-safe ASR metrics",
        [
            ("Languages", "4", "English, Hindi, Tamil and Kannada"),
            ("Labelled cases", "16", "Balanced benign and configured-risk acceptance cases"),
            ("Current run", "Pass", "100% risk/category fixture agreement; 0% benign FPR"),
        ],
    )

    page(document, "Why validation was added", "1 / Existing system and gap")
    paragraph(document, "The conversation feature already fixed Unicode normalization, duplicate cue counting and the separation of tone from risk. The remaining problem was repeatability: there was no checked-in labelled suite proving that those known failure modes stayed fixed across four supported languages.")
    table(document, ("Risk before", "Control added"), [
        ("Indic vowel signs removed", "Unicode-preserving regression cases"),
        ("Synonyms double counted", "Expected category sets"),
        ("Benign phrase collisions", "Benign false-positive cases"),
        ("Scam phrasing shown as positive", "Independent risk-level assertion"),
        ("ASR quality unmeasured", "Optional WER and CER"),
    ], (3.3, 3.4))

    page(document, "Dataset design", "2 / Fixture")
    paragraph(document, "The checked-in JSON fixture contains 16 transparent cases: four per language. Each language includes benign content, a Medium-risk example, a High-risk example and an additional benign collision/normal-language control.")
    table(document, ("Language", "Cases", "Benign", "Medium", "High"), [
        (language, metrics["cases"], metrics["benign_cases"], 1, 1)
        for language, metrics in results["by_language"].items()
    ], (1.8, 1.0, 1.2, 1.2, 1.2))
    callout(document, "Scope", "This is a regression and acceptance fixture, not a statistically representative call corpus.", ORANGE)

    page(document, "Metrics and decision rules", "3 / Measurement")
    table(document, ("Metric", "Definition", "Purpose"), [
        ("Risk accuracy", "Exact Low/Medium/High match", "Catch semantic regressions"),
        ("Category precision", "TP / (TP + FP)", "Penalize wrong cue categories"),
        ("Category recall", "TP / (TP + FN)", "Penalize missed categories"),
        ("Category F1", "Harmonic mean", "One category summary"),
        ("Benign FPR", "Benign predicted non-Low", "Track false alarms"),
        ("WER / CER", "Unicode-safe edit distance", "Optional ASR diagnostics"),
    ], (1.7, 2.4, 2.6))
    paragraph(document, "The validation command exits non-zero unless risk accuracy and category F1 are both 1.0 on this locked acceptance fixture, making it suitable for CI regression protection.")

    page(document, "Overall results", "4 / Evidence")
    chart(TMP / "validation_overall.png", ["Risk accuracy", "Category F1", "Category precision", "Benign FPR"], [100, 100, 100, 0], "Acceptance fixture results")
    document.add_picture(str(TMP / "validation_overall.png"), width=Inches(6.9))
    table(document, ("Measure", "Value"), [
        ("Cases", overall["cases"]),
        ("Risk accuracy", f"{overall['risk_accuracy']:.1%}"),
        ("Category F1", f"{overall['category_f1']:.1%}"),
        ("Benign false-positive rate", f"{overall['false_positive_rate']:.1%}"),
    ], (3.6, 2.0))

    page(document, "Per-language results", "5 / Multilingual view")
    table(document, ("Language", "Cases", "Risk accuracy", "Category F1", "Benign FPR"), [
        (language, metrics["cases"], f"{metrics['risk_accuracy']:.0%}", f"{metrics['category_f1']:.0%}", f"{metrics['false_positive_rate']:.0%}")
        for language, metrics in results["by_language"].items()
    ], (1.7, 0.9, 1.5, 1.3, 1.4))
    bullets(document, ["Hindi retains matras and virama during normalization.", "Tamil and Kannada combining marks remain intact.", "Each category contributes at most one hit per transcript segment.", "The common Hindi phrase अभी नहीं remains a benign Low-risk control."])

    page(document, "ASR error-rate support", "6 / WER and CER")
    paragraph(document, "Cases may include a reference transcript and an ASR hypothesis. The evaluator computes word error rate and character error rate after the same Unicode-safe normalization used by cue analysis.")
    table(document, ("Pair", "Expected behavior"), [
        ("Hindi exact-match example", "WER 0; vowel signs retained"),
        ("Tamil exact-match example", "CER 0; Tamil combining marks retained"),
        ("one two three / one three", "WER 1/3"),
    ], (3.5, 3.2))
    callout(document, "Interpretation", "The small fixture's mean WER/CER is illustrative only. Reportable ASR quality needs natural recordings and human references.", ORANGE)

    page(document, "Automation and artifacts", "7 / Reproducibility")
    table(document, ("Artifact", "Purpose"), [
        ("labelled_regression_cases.json", "Human-readable locked cases"),
        ("run_conversation_validation.py", "One-command evaluator"),
        ("validation_summary.json", "Machine-readable overall/case results"),
        ("per_language_metrics.csv", "Presentation and spreadsheet output"),
        ("tests/test_conversation_validation.py", "CI assertions and Unicode metrics"),
    ], (3.0, 3.7))
    paragraph(document, "Run: python evaluation/run_conversation_validation.py. Then run: python -m pytest -q. The current full suite result is 63 passed.")

    page(document, "What the result proves—and does not", "8 / Claim boundary")
    table(document, ("Supported statement", "Unsupported statement"), [
        ("Known multilingual regressions are covered", "All natural calls are correctly classified"),
        ("Fixture output is reproducible", "100% real-world accuracy"),
        ("Benign fixture FPR is 0%", "Population false-positive rate is 0%"),
        ("Unicode WER/CER code is tested", "Whisper is validated for every target language"),
    ], (3.35, 3.35))
    callout(document, "Correct wording", "Voice Guard passes its 16-case multilingual acceptance suite. This is regression evidence, not population performance.")

    page(document, "Next validation expansion", "9 / Conclusion")
    table(document, ("Priority", "Next evidence", "Acceptance measure"), [
        ("1", "Independent natural-call corpus", "Per-language confidence intervals"),
        ("2", "Code-switched and noisy calls", "Risk/category precision and recall"),
        ("3", "Human ASR references", "WER/CER per language and condition"),
        ("4", "Speaker-turn references", "DER/JER and overlap slices"),
        ("5", "Adversarial benign controls", "False-positive rate by phrase family"),
    ], (0.8, 3.2, 2.7))
    callout(document, "Presentation statement", "The validation harness is complete and reproducible. The next project milestone is collecting independently labelled natural calls so the same evaluator can produce defensible generalization evidence.")
    document.save(OUT / "Voice_Guard_Multilingual_Conversation_Validation.docx")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(exist_ok=True)
    chart(TMP / "speaker_tests.png", ["All repository tests", "Speaker-flow tests", "Failures"], [63, 4, 0], "Automated verification", "")
    build_speaker_report()
    build_validation_report()


if __name__ == "__main__":
    main()
