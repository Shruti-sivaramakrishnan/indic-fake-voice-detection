"""Dependency-light analysis of timestamped call transcripts.

This module intentionally does not perform speech recognition. It accepts an
optional transcript exported by a human or ASR system and surfaces lexical
conversation cues for review. Cue hits are not evidence of fraud or intent.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


MAX_TRANSCRIPT_BYTES = 2 * 1024 * 1024
MAX_SEGMENTS = 5_000


@dataclass(frozen=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True)
class CueHit:
    start_seconds: float
    end_seconds: float
    category: str
    matched_phrase: str
    text: str


@dataclass(frozen=True)
class ConversationReport:
    segments: int
    cue_hits: tuple[CueHit, ...]
    category_counts: dict[str, int]
    sentiment: str
    positive_hits: int
    negative_hits: int
    first_elevated_cue_time: float | None


CUE_PHRASES = {
    "urgency": (
        "urgent", "immediately", "right now", "hurry", "last chance",
        "तुरंत", "अभी", "जल्दी", "உடனே", "இப்போது", "ತಕ್ಷಣ", "ಈಗಲೇ",
    ),
    "secrecy": (
        "do not tell", "don't tell", "keep this secret", "confidential",
        "किसी को मत बताना", "ரகசியம்", "யாரிடமும் சொல்லாதே",
        "ರಹಸ್ಯ", "ಯಾರಿಗೂ ಹೇಳಬೇಡಿ",
    ),
    "money_or_payment": (
        "send money", "transfer money", "bank account", "payment", "upi",
        "पैसे भेज", "पैसा भेज", "भुगतान", "பணம் அனுப்பு", "வங்கி கணக்கு",
        "ಹಣ ಕಳುಹಿಸಿ", "ಬ್ಯಾಂಕ್ ಖಾತೆ",
    ),
    "credential_or_otp": (
        "otp", "one time password", "pin number", "verification code",
        "ओटीपी", "पासवर्ड", "ஓடிபி", "கடவுச்சொல்", "ಒಟಿಪಿ", "ಪಾಸ್‌ವರ್ಡ್",
    ),
    "authority_or_familiarity": (
        "this is your bank", "police officer", "your manager", "your boss",
        "i am your friend", "मैं बैंक से", "पुलिस", "நான் வங்கியிலிருந்து",
        "காவல்துறை", "ನಾನು ಬ್ಯಾಂಕಿನಿಂದ", "ಪೊಲೀಸ್",
    ),
    "pressure_or_threat": (
        "account will be blocked", "you will be arrested", "penalty",
        "legal action", "खाता बंद", "गिरफ्तार", "அக்கவுண்ட் முடக்கப்படும்",
        "கைது", "ಖಾತೆ ನಿರ್ಬಂಧ", "ಬಂಧಿಸಲಾಗುವುದು",
    ),
}

POSITIVE_PHRASES = (
    "thank you", "thanks", "good", "happy", "safe", "धन्यवाद", "अच्छा",
    "நன்றி", "நல்ல", "ಧನ್ಯವಾದ", "ಒಳ್ಳೆಯ",
)
NEGATIVE_PHRASES = (
    "afraid", "angry", "worried", "problem", "danger", "scared", "डर",
    "गुस्सा", "चिंता", "பயம்", "கோபம்", "கவலை", "ಭಯ", "ಕೋಪ", "ಚಿಂತೆ",
)


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"[^\w\s'‌-]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def parse_transcript_csv(
    payload: bytes, call_duration_seconds: float | None = None
) -> list[TranscriptSegment]:
    """Parse ``start_seconds,end_seconds,text`` CSV data with strict limits."""
    if len(payload) > MAX_TRANSCRIPT_BYTES:
        raise ValueError("Transcript CSV is larger than 2 MB.")
    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Transcript CSV must use UTF-8 encoding.") from exc

    reader = csv.DictReader(io.StringIO(decoded))
    required = {"start_seconds", "end_seconds", "text"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("CSV needs start_seconds, end_seconds, and text columns.")

    segments = []
    for row_number, row in enumerate(reader, start=2):
        if len(segments) >= MAX_SEGMENTS:
            raise ValueError(f"Transcript exceeds the {MAX_SEGMENTS:,}-segment limit.")
        try:
            start = float(row["start_seconds"])
            end = float(row["end_seconds"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Row {row_number} has invalid timestamps.") from exc
        text = (row.get("text") or "").strip()
        if start < 0 or end <= start or not text:
            raise ValueError(f"Row {row_number} has an invalid interval or empty text.")
        if call_duration_seconds is not None and end > call_duration_seconds + 1.0:
            raise ValueError(f"Row {row_number} extends beyond the call duration.")
        segments.append(TranscriptSegment(start, end, text))

    if not segments:
        raise ValueError("Transcript CSV contains no segments.")
    if any(
        current.start_seconds < previous.start_seconds
        for previous, current in zip(segments, segments[1:])
    ):
        raise ValueError("Transcript segments must be sorted by start_seconds.")
    return segments


def _phrase_hits(normalized: str, phrases: Iterable[str]) -> list[str]:
    padded = f" {normalized} "
    return [phrase for phrase in phrases if f" {_normalize(phrase)} " in padded]


def analyze_transcript(segments: Iterable[TranscriptSegment]) -> ConversationReport:
    segment_list = list(segments)
    hits = []
    positive_hits = 0
    negative_hits = 0
    elevated_time = None

    for segment in segment_list:
        normalized = _normalize(segment.text)
        segment_categories = set()
        for category, phrases in CUE_PHRASES.items():
            for phrase in _phrase_hits(normalized, phrases):
                hits.append(
                    CueHit(
                        segment.start_seconds,
                        segment.end_seconds,
                        category,
                        phrase,
                        segment.text,
                    )
                )
                segment_categories.add(category)
        positive_hits += len(_phrase_hits(normalized, POSITIVE_PHRASES))
        negative_hits += len(_phrase_hits(normalized, NEGATIVE_PHRASES))
        if elevated_time is None and len(segment_categories) >= 2:
            elevated_time = segment.start_seconds

    if positive_hits and negative_hits:
        sentiment = "mixed"
    elif positive_hits:
        sentiment = "positive"
    elif negative_hits:
        sentiment = "negative"
    else:
        sentiment = "neutral_or_unknown"

    counts = Counter(hit.category for hit in hits)
    return ConversationReport(
        segments=len(segment_list),
        cue_hits=tuple(hits),
        category_counts=dict(sorted(counts.items())),
        sentiment=sentiment,
        positive_hits=positive_hits,
        negative_hits=negative_hits,
        first_elevated_cue_time=elevated_time,
    )
