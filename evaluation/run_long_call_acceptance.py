"""Run controlled multilingual long-call acceptance tests with the v1 checkpoint.

The source clips are frozen-test examples fetched by dataset row index. They are
used only to confirm the already-designed long-call plumbing; this script does
not select thresholds or alter the model. Generated calls stay in memory and
only aggregate results are written to disk.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import call_analysis
import model_pipeline


DATASET = "SherryT997/IndicTTS-Deepfake-Challenge-Data"
TARGET_SR = 16_000
TRUE_FAKE_START = 120.0
TRUE_FAKE_END = 180.0


@dataclass(frozen=True)
class CallSpec:
    call_id: str
    language: str
    real_indices: tuple[int, ...]
    fake_indices: tuple[int, ...]
    rotation: int = 0


CALL_SPECS = (
    CallSpec("assamese_1", "Assamese", (24, 25, 33), (0, 54, 62), 0),
    CallSpec("assamese_2", "Assamese", (24, 25, 33), (0, 54, 62), 1),
    CallSpec("bengali_1", "Bengali", (2006, 2038, 2073), (2014, 2056, 2063), 0),
    CallSpec("bengali_2", "Bengali", (2006, 2038, 2073), (2014, 2056, 2063), 1),
    CallSpec("bodo_1", "Bodo", (3816, 3823, 3897), (3820, 3834, 3850), 0),
)


def dataset_row(index: int) -> dict:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": index,
            "length": 1,
        }
    )
    payload = fetch_json(
        f"https://datasets-server.huggingface.co/rows?{query}", timeout=120
    )
    rows = payload.get("rows") or []
    if len(rows) != 1 or rows[0].get("row_idx") != index:
        raise RuntimeError(f"Dataset server did not return row {index}.")
    return rows[0]["row"]


def fetch_bytes(url: str, timeout: int, attempts: int = 4) -> bytes:
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "VoiceGuardAcceptance/1.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (OSError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Download failed after {attempts} attempts: {last_error}")


def fetch_json(url: str, timeout: int) -> dict:
    return json.loads(fetch_bytes(url, timeout=timeout).decode("utf-8"))


def fetch_clip(index: int, expected_language: str, expected_label: int) -> np.ndarray:
    row = dataset_row(index)
    if row["language"] != expected_language or int(row["is_tts"]) != expected_label:
        raise RuntimeError(
            f"Row {index} metadata changed: {row['language']!r}, {row['is_tts']!r}."
        )
    sources = row.get("audio") or []
    if not sources or not sources[0].get("src"):
        raise RuntimeError(f"Row {index} has no downloadable audio asset.")
    audio_bytes = fetch_bytes(sources[0]["src"], timeout=180)
    waveform, sample_rate = model_pipeline.decode_audio_bytes(audio_bytes)
    waveform = model_pipeline.ensure_target_sample_rate(
        np.asarray(waveform, dtype=np.float32), sample_rate, TARGET_SR
    )
    return np.asarray(waveform, dtype=np.float32).reshape(-1)


def rotate(items: list[np.ndarray], amount: int) -> list[np.ndarray]:
    if not items:
        raise ValueError("At least one clip is required.")
    offset = amount % len(items)
    return items[offset:] + items[:offset]


def fill_duration(clips: list[np.ndarray], seconds: float) -> np.ndarray:
    target_samples = int(round(seconds * TARGET_SR))
    if target_samples <= 0:
        raise ValueError("Duration must be positive.")
    parts = []
    total = 0
    index = 0
    while total < target_samples:
        clip = clips[index % len(clips)]
        remaining = target_samples - total
        part = clip[:remaining]
        parts.append(part)
        total += part.size
        index += 1
    return np.concatenate(parts).astype(np.float32, copy=False)


def wav_bytes(waveform: np.ndarray) -> bytes:
    clipped = np.clip(waveform, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(TARGET_SR)
        stream.writeframes(pcm.tobytes())
    return output.getvalue()


def best_flagged_range(report: call_analysis.CallAnalysisReport):
    best = None
    best_overlap = -1.0
    for region in report.flagged_ranges:
        overlap = max(
            0.0,
            min(region.end_seconds, TRUE_FAKE_END)
            - max(region.start_seconds, TRUE_FAKE_START),
        )
        if overlap > best_overlap:
            best = region
            best_overlap = overlap
    return best, best_overlap


def run(output_dir: Path) -> dict:
    required = sorted(
        {
            (index, spec.language, label)
            for spec in CALL_SPECS
            for label, indices in ((0, spec.real_indices), (1, spec.fake_indices))
            for index in indices
        }
    )
    clips = {}
    for position, (index, language, label) in enumerate(required, start=1):
        print(f"Fetching {position}/{len(required)}: row {index} ({language}, {label})")
        clips[index] = fetch_clip(index, language, label)

    # Load once before timing individual calls.
    model_pipeline.load_inference_bundle()
    model_pipeline.load_frontend()

    spliced_results = []
    all_real_results = []
    for spec in CALL_SPECS:
        real = rotate([clips[index] for index in spec.real_indices], spec.rotation)
        fake = rotate([clips[index] for index in spec.fake_indices], spec.rotation)
        first_real = fill_duration(real, 120.0)
        fake_middle = fill_duration(fake, 60.0)
        second_real = fill_duration(rotate(real, 1), 120.0)
        spliced = np.concatenate([first_real, fake_middle, second_real])

        started = time.perf_counter()
        report = call_analysis.analyze_call(wav_bytes(spliced))
        elapsed = time.perf_counter() - started
        best, overlap = best_flagged_range(report)
        start_delta = None if best is None else best.start_seconds - TRUE_FAKE_START
        end_delta = None if best is None else best.end_seconds - TRUE_FAKE_END
        boundary_pass = bool(
            best is not None
            and abs(start_delta) <= 6.0
            and abs(end_delta) <= 6.0
        )
        spliced_results.append(
            {
                "call_id": spec.call_id,
                "language": spec.language,
                "windows": report.windows_analysed,
                "flagged_share": report.flagged_share,
                "best_flagged_start": None if best is None else best.start_seconds,
                "best_flagged_end": None if best is None else best.end_seconds,
                "start_delta_seconds": start_delta,
                "end_delta_seconds": end_delta,
                "overlap_seconds": overlap,
                "boundary_within_6_seconds": boundary_pass,
                "elapsed_seconds": elapsed,
            }
        )

        all_real = fill_duration(real, 300.0)
        started = time.perf_counter()
        real_report = call_analysis.analyze_call(wav_bytes(all_real))
        all_real_results.append(
            {
                "call_id": spec.call_id.replace("_", "_all_real_", 1),
                "language": spec.language,
                "windows": real_report.windows_analysed,
                "flagged_windows": real_report.flagged_windows,
                "flagged_share": real_report.flagged_share,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )

    payload = {
        "protocol": {
            "source": "historical frozen v1 test examples",
            "purpose": "feature acceptance only; no tuning or final generalisation claim",
            "spliced_layout_seconds": [120, 60, 120],
            "expected_fake_interval_seconds": [TRUE_FAKE_START, TRUE_FAKE_END],
            "calls": len(CALL_SPECS),
            "languages": sorted({spec.language for spec in CALL_SPECS}),
        },
        "spliced_results": spliced_results,
        "all_real_results": all_real_results,
        "summary": {
            "spliced_boundary_passes": sum(
                row["boundary_within_6_seconds"] for row in spliced_results
            ),
            "spliced_calls": len(spliced_results),
            "mean_all_real_flagged_share": float(
                np.mean([row["flagged_share"] for row in all_real_results])
            ),
            "max_all_real_flagged_share": max(
                row["flagged_share"] for row in all_real_results
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "long_call_acceptance.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/long_call_acceptance"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
