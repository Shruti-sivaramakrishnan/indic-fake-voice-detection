"""Transmission-condition robustness analysis for Voice Guard.

This module deliberately keeps FFmpeg orchestration separate from the model
pipeline.  It creates deterministic, in-memory audio variants and sends each
variant through the same public analysis callable used by the Streamlit app.
No transformed audio is written to disk.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple


class CodecRobustnessError(RuntimeError):
    """Safe, user-displayable failure of the robustness analysis."""


@dataclass(frozen=True)
class CodecCondition:
    """One reproducible transmission condition implemented with FFmpeg."""

    condition_id: str
    label: str
    technical_summary: str
    ffmpeg_args: Tuple[str, ...]


DEFAULT_CONDITIONS: Tuple[CodecCondition, ...] = (
    CodecCondition(
        condition_id="opus_16kbps",
        label="Opus 16 kbps",
        technical_summary="Mono Opus, 16 kbps, VoIP mode",
        ffmpeg_args=(
            "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "16k",
            "-vbr", "on", "-application", "voip", "-f", "ogg",
        ),
    ),
    CodecCondition(
        condition_id="g711_alaw_8khz",
        label="G.711 A-law · 8 kHz",
        technical_summary="Mono G.711 A-law with telephone-band filtering",
        ffmpeg_args=(
            "-ac", "1", "-ar", "8000", "-af", "highpass=f=300,lowpass=f=3400",
            "-c:a", "pcm_alaw", "-f", "wav",
        ),
    ),
    CodecCondition(
        condition_id="telephone_pcm_8khz",
        label="Telephone band · 8 kHz",
        technical_summary="Mono 16-bit PCM, 8 kHz, 300–3400 Hz bandwidth",
        ffmpeg_args=(
            "-ac", "1", "-ar", "8000", "-af", "highpass=f=300,lowpass=f=3400",
            "-c:a", "pcm_s16le", "-f", "wav",
        ),
    ),
)


@dataclass(frozen=True)
class ConditionResult:
    """Model result and stability measurements for one condition."""

    condition_id: str
    label: str
    technical_summary: str
    verdict: Optional[str]
    p_fake: Optional[float]
    delta_p_fake: Optional[float]
    verdict_changed: Optional[bool]
    sample_rate: Optional[int]
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class CodecRobustnessReport:
    """Original result plus every requested simulated transmission condition."""

    source_hash: str
    original_verdict: str
    original_p_fake: float
    conditions: Tuple[ConditionResult, ...]

    @property
    def successful_conditions(self) -> Tuple[ConditionResult, ...]:
        return tuple(condition for condition in self.conditions if condition.succeeded)

    @property
    def failed_conditions(self) -> Tuple[ConditionResult, ...]:
        return tuple(condition for condition in self.conditions if not condition.succeeded)

    @property
    def transformed_success_count(self) -> int:
        return sum(
            condition.succeeded and condition.condition_id != "original"
            for condition in self.conditions
        )

    @property
    def verdict_change_count(self) -> int:
        return sum(condition.verdict_changed is True for condition in self.conditions)

    @property
    def max_absolute_delta(self) -> float:
        deltas = [
            abs(condition.delta_p_fake)
            for condition in self.successful_conditions
            if condition.delta_p_fake is not None
        ]
        return max(deltas, default=0.0)


def find_ffmpeg() -> str:
    """Return the FFmpeg executable or raise a safe setup error."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise CodecRobustnessError(
            "FFmpeg is required for the transmission robustness check but is not "
            "available in this environment."
        )
    return executable


def transcode_audio(
    audio_bytes: bytes,
    condition: CodecCondition,
    *,
    ffmpeg_path: Optional[str] = None,
    timeout_seconds: float = 60.0,
) -> bytes:
    """Apply one declared condition and return encoded bytes without disk I/O."""
    if not audio_bytes:
        raise CodecRobustnessError("The source audio is empty.")

    executable = ffmpeg_path or find_ffmpeg()
    command = [
        executable,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-map_metadata",
        "-1",
        "-vn",
        *condition.ffmpeg_args,
        "pipe:1",
    ]

    try:
        completed = subprocess.run(
            command,
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise CodecRobustnessError(
            f"Timed out while creating the {condition.label} condition."
        ) from exc
    except OSError as exc:
        raise CodecRobustnessError(
            f"Could not start FFmpeg for the {condition.label} condition."
        ) from exc

    if completed.returncode != 0 or not completed.stdout:
        raise CodecRobustnessError(
            f"FFmpeg could not create the {condition.label} condition."
        )
    return completed.stdout


def _p_fake(result: Any) -> float:
    try:
        value = float(result.metadata["p_fake_calibrated"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise CodecRobustnessError(
            "The model result did not include a calibrated AI-likelihood score."
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise CodecRobustnessError(
            "The model returned an invalid AI-likelihood score."
        )
    return value


def _sample_rate(result: Any) -> Optional[int]:
    try:
        value = result.metadata.get("sample_rate_in")
        return int(value) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def analyze_codec_robustness(
    audio_bytes: bytes,
    analyze_fn: Callable[[bytes], Any],
    *,
    original_result: Optional[Any] = None,
    conditions: Sequence[CodecCondition] = DEFAULT_CONDITIONS,
    ffmpeg_path: Optional[str] = None,
) -> CodecRobustnessReport:
    """Score original and simulated conditions through one shared model API.

    A failed variant is recorded without discarding successful rows.  The
    original analysis is mandatory because every delta and verdict-change flag
    is relative to it.
    """
    if not audio_bytes:
        raise CodecRobustnessError("The source audio is empty.")

    original = original_result if original_result is not None else analyze_fn(audio_bytes)
    original_p_fake = _p_fake(original)
    original_verdict = str(original.verdict)
    source_hash = hashlib.sha256(audio_bytes).hexdigest()
    result_hash = getattr(original, "audio_hash", source_hash)
    if result_hash != source_hash:
        raise CodecRobustnessError(
            "The existing model result belongs to a different audio file."
        )

    executable = ffmpeg_path or find_ffmpeg()
    rows = [
        ConditionResult(
            condition_id="original",
            label="Original",
            technical_summary="Uploaded audio without transmission simulation",
            verdict=original_verdict,
            p_fake=original_p_fake,
            delta_p_fake=0.0,
            verdict_changed=False,
            sample_rate=_sample_rate(original),
        )
    ]

    for condition in conditions:
        try:
            transformed = transcode_audio(
                audio_bytes,
                condition,
                ffmpeg_path=executable,
            )
            analysis = analyze_fn(transformed)
            p_fake = _p_fake(analysis)
            verdict = str(analysis.verdict)
            rows.append(
                ConditionResult(
                    condition_id=condition.condition_id,
                    label=condition.label,
                    technical_summary=condition.technical_summary,
                    verdict=verdict,
                    p_fake=p_fake,
                    delta_p_fake=p_fake - original_p_fake,
                    verdict_changed=verdict != original_verdict,
                    sample_rate=_sample_rate(analysis),
                )
            )
        except Exception as exc:  # keep other successful conditions usable
            message = str(exc) if isinstance(exc, CodecRobustnessError) else (
                f"Analysis failed for the {condition.label} condition."
            )
            rows.append(
                ConditionResult(
                    condition_id=condition.condition_id,
                    label=condition.label,
                    technical_summary=condition.technical_summary,
                    verdict=None,
                    p_fake=None,
                    delta_p_fake=None,
                    verdict_changed=None,
                    sample_rate=None,
                    error=message,
                )
            )

    return CodecRobustnessReport(
        source_hash=source_hash,
        original_verdict=original_verdict,
        original_p_fake=original_p_fake,
        conditions=tuple(rows),
    )
