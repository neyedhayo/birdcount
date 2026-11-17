#!/usr/bin/env python3
"""Embed time-annotated Ovenbird vocalizations without rerunning detection.

The output array and metadata table are aligned by ``embedding_row``. The
metadata table retains every source annotation, including rows whose audio file
is missing. Only rows with ``embedding_status == "embedded"`` have an array row.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints/full_2025-04-10T11:02:36.028451_best.pth"
)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
WINDOW_DURATION_SECONDS = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_recordings(recordings_dir: Path) -> dict[str, Path]:
    """Map filename stems to audio paths, rejecting ambiguous duplicate stems."""
    paths = sorted(
        path
        for path in recordings_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    mapping: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in paths:
        if path.stem in mapping:
            duplicates.add(path.stem)
        mapping[path.stem] = path.resolve()
    if duplicates:
        raise ValueError(
            "Multiple audio files have the same filename stem: "
            f"{sorted(duplicates)}"
        )
    return mapping


def prepare_metadata(
    annotations: pd.DataFrame,
    recording_paths: dict[str, Path],
) -> pd.DataFrame:
    """Add paths and deterministic, clamped two-second embedding windows."""
    required = {
        "recording_id",
        "recording_duration",
        "detection_time",
        "tag_duration",
    }
    missing = required.difference(annotations.columns)
    if missing:
        raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")

    metadata = annotations.copy()
    metadata.insert(0, "source_row", np.arange(len(metadata), dtype=int))
    recording_keys = metadata["recording_id"].astype(str)
    metadata["file"] = recording_keys.map(
        lambda key: str(recording_paths[key]) if key in recording_paths else ""
    )
    metadata["song_center_time"] = (
        metadata["detection_time"] + metadata["tag_duration"] / 2
    )
    metadata["window_start_raw"] = metadata["song_center_time"] - (
        WINDOW_DURATION_SECONDS / 2
    )
    latest_start = metadata["recording_duration"] - WINDOW_DURATION_SECONDS
    if (latest_start < 0).any():
        bad_ids = metadata.loc[latest_start < 0, "recording_id"].unique().tolist()
        raise ValueError(
            "Recordings shorter than the two-second model window: "
            f"{bad_ids[:10]}"
        )
    metadata["start_time"] = np.minimum(
        np.maximum(metadata["window_start_raw"], 0),
        latest_start,
    )
    metadata["end_time"] = metadata["start_time"] + WINDOW_DURATION_SECONDS
    metadata["boundary_adjusted"] = ~np.isclose(
        metadata["start_time"], metadata["window_start_raw"]
    )
    time_columns = [
        "song_center_time",
        "window_start_raw",
        "start_time",
        "end_time",
    ]
    metadata[time_columns] = metadata[time_columns].round(6)
    metadata["embedding_status"] = np.where(
        metadata["file"].ne(""), "pending", "missing_audio"
    )
    metadata["embedding_row"] = pd.Series(pd.NA, index=metadata.index, dtype="Int64")
    return metadata


def embed_available_rows(
    metadata: pd.DataFrame,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    """Embed available annotations and return array plus finalized metadata."""
    available_mask = metadata["embedding_status"].eq("pending")
    available = metadata.loc[available_mask].copy()
    if available.empty:
        raise ValueError("No annotation rows have a matching audio file.")

    # OpenSoundscape treats remaining columns as target labels. Inference needs
    # only the clip index; retain all metadata separately for the output join.
    clips = available.set_index(["file", "start_time", "end_time"])[[]]
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from model import load_ovenbird_model

    model = load_ovenbird_model()
    print(f"Embedding {len(clips):,} annotations on {model.device}...")
    result = model.embed(
        clips,
        batch_size=batch_size,
        num_workers=num_workers,
        progress_bar=True,
    )
    embeddings = result.values if isinstance(result, pd.DataFrame) else np.asarray(result)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(available):
        raise RuntimeError(
            "Unexpected embedding shape: "
            f"{embeddings.shape}; expected ({len(available)}, n_features)"
        )
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Embedding array contains NaN or infinite values.")

    finalized = metadata.copy()
    finalized.loc[available_mask, "embedding_status"] = "embedded"
    finalized.loc[available_mask, "embedding_row"] = np.arange(
        len(available), dtype=int
    )
    return embeddings, finalized, str(model.device)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Ovenbird embeddings from an annotation CSV and recordings."
    )
    parser.add_argument("annotations_csv", type=Path)
    parser.add_argument("recordings_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations_path = args.annotations_csv.expanduser().resolve()
    recordings_dir = args.recordings_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {annotations_path}")
    if not recordings_dir.is_dir():
        raise NotADirectoryError(f"Recordings directory not found: {recordings_dir}")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and num workers cannot be negative")

    annotations = pd.read_csv(annotations_path)
    recording_paths = discover_recordings(recordings_dir)
    metadata = prepare_metadata(annotations, recording_paths)
    embeddings, metadata, device = embed_available_rows(
        metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "ovenbird_embeddings.npy"
    metadata_path = output_dir / "ovenbird_embedding_metadata.csv"
    manifest_path = output_dir / "ovenbird_embedding_manifest.json"

    np.save(embeddings_path, embeddings, allow_pickle=False)
    metadata.to_csv(metadata_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": annotations_path.name,
        "source_csv_sha256": sha256(annotations_path),
        "recordings_directory": recordings_dir.name,
        "checkpoint": CHECKPOINT_PATH.name,
        "checkpoint_sha256": sha256(CHECKPOINT_PATH),
        "device": device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "window_duration_seconds": WINDOW_DURATION_SECONDS,
        "window_rule": "center detection_time + tag_duration/2; clamp to recording bounds",
        "source_rows": int(len(metadata)),
        "embedded_rows": int(metadata["embedding_status"].eq("embedded").sum()),
        "missing_audio_rows": int(
            metadata["embedding_status"].eq("missing_audio").sum()
        ),
        "missing_recording_ids": sorted(
            metadata.loc[
                metadata["embedding_status"].eq("missing_audio"), "recording_id"
            ].unique().tolist()
        ),
        "boundary_adjusted_rows": int(metadata["boundary_adjusted"].sum()),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "embeddings_file": embeddings_path.name,
        "metadata_file": metadata_path.name,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "opensoundscape": package_version("opensoundscape"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Saved embeddings: {embeddings_path} {embeddings.shape} {embeddings.dtype}")
    print(f"Saved metadata:   {metadata_path} ({len(metadata):,} rows)")
    print(f"Saved manifest:   {manifest_path}")
    if manifest["missing_recording_ids"]:
        print(
            "Missing audio for recording IDs: "
            + ", ".join(str(value) for value in manifest["missing_recording_ids"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
