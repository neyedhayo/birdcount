#!/usr/bin/env python3
"""Estimate candidate Ovenbird individuals in one long audio recording.

This is an inference workflow, not a training workflow:

1. HawkEars scores overlapping three-second windows for Ovenbird presence.
2. Temporal non-maximum suppression removes duplicate overlapping detections.
3. The included Ovenbird checkpoint embeds a centered two-second window for
   each retained detection.
4. UMAP and HDBSCAN group repeated songs into candidate individuals.

The result is an estimate of acoustically distinct, repeatedly singing birds.
It is not a direct count of every bird physically present in the recording.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"


def select_separated_detections(
    detections: pd.DataFrame,
    min_separation_seconds: float,
) -> pd.DataFrame:
    """Keep the highest-scoring detection within each temporal neighborhood."""
    required = {"start_time", "end_time", "score"}
    missing = required.difference(detections.columns)
    if missing:
        raise ValueError(f"detections table is missing columns: {sorted(missing)}")

    candidates = detections.copy()
    candidates["center_time"] = (
        candidates["start_time"].astype(float)
        + candidates["end_time"].astype(float)
    ) / 2
    candidates = candidates.sort_values("score", ascending=False)

    selected_rows = []
    selected_centers: list[float] = []
    for _, row in candidates.iterrows():
        center = float(row["center_time"])
        if all(
            abs(center - existing) >= min_separation_seconds
            for existing in selected_centers
        ):
            selected_rows.append(row)
            selected_centers.append(center)

    if not selected_rows:
        return candidates.iloc[0:0].copy()
    return (
        pd.DataFrame(selected_rows)
        .sort_values("center_time")
        .reset_index(drop=True)
    )


def detect_ovenbird_windows(
    audio_path: Path,
    threshold: float,
    min_separation_seconds: float,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """Run HawkEars on overlapping windows and retain distinct song events."""
    try:
        import bioacoustics_model_zoo as bmz
    except ImportError as exc:
        raise RuntimeError(
            "HawkEars is not installed. Create the Python 3.11 environment and "
            "install the inference requirements as described in README.md."
        ) from exc

    detector = bmz.HawkEars()
    scores = detector.predict(
        [str(audio_path)],
        batch_size=batch_size,
        num_workers=num_workers,
        clip_overlap_fraction=2 / 3,
        final_clip="extend",
    )
    if "Ovenbird" not in scores.columns:
        raise RuntimeError(
            "HawkEars did not return an 'Ovenbird' column. Available columns "
            f"include: {list(scores.columns[:10])}"
        )

    candidates = scores[["Ovenbird"]].reset_index()
    candidates = candidates.rename(columns={"Ovenbird": "score"})
    candidates = candidates[candidates["score"] > threshold].copy()
    return select_separated_detections(candidates, min_separation_seconds)


def load_detection_table(
    csv_path: Path,
    audio_path: Path,
    threshold: float,
    min_separation_seconds: float,
) -> pd.DataFrame:
    """Load precomputed/manual song windows instead of running HawkEars."""
    detections = pd.read_csv(csv_path)
    if "score" not in detections:
        detections["score"] = 1.0
    if "file" not in detections:
        detections["file"] = str(audio_path)
    detections = detections[detections["score"] > threshold].copy()
    return select_separated_detections(detections, min_separation_seconds)


def make_embedding_windows(
    detections: pd.DataFrame,
    audio_path: Path,
) -> pd.DataFrame:
    """Create centered two-second windows expected by the AIID checkpoint."""
    windows = detections.copy()
    windows["file"] = str(audio_path)
    windows["start_time"] = np.maximum(windows["center_time"] - 1.0, 0.0)
    windows["end_time"] = windows["start_time"] + 2.0
    return windows.set_index(["file", "start_time", "end_time"])


def embed_song_windows(
    windows: pd.DataFrame,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Generate one 512-dimensional embedding for each song window."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    try:
        from model import load_ovenbird_model
    except ImportError as exc:
        raise RuntimeError(
            "The Ovenbird model dependencies are not installed. Create the Python "
            "3.11 environment and install the inference requirements as described "
            "in README.md."
        ) from exc

    model = load_ovenbird_model()
    embeddings = model.embed(
        windows,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if isinstance(embeddings, pd.DataFrame):
        embeddings = embeddings.values
    return np.asarray(embeddings)


def cluster_song_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce embeddings with UMAP and cluster them with HDBSCAN."""
    if len(embeddings) < min_cluster_size:
        return np.full(len(embeddings), -1, dtype=int), np.empty((len(embeddings), 0))

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    import evaluation

    # UMAP's spectral initialization needs more samples than output dimensions.
    # For very small sets, cluster the standardized 512-D embeddings directly.
    use_reduction = len(embeddings) >= 5
    n_dimensions = min(5, len(embeddings) - 3) if use_reduction else None
    labels, reduced = evaluation.cluster(
        pd.DataFrame(index=range(len(embeddings))),
        embeddings,
        reduction_algorithm="umap" if use_reduction else None,
        reduced_n_dimensions=n_dimensions,
        min_cluster_size=min_cluster_size,
        random_state=random_state,
        allow_single_cluster=True,
    )
    return np.asarray(labels, dtype=int), np.asarray(reduced)


def summarize(labels: np.ndarray, n_songs: int, min_cluster_size: int) -> dict:
    """Create a conservative, machine-readable result summary."""
    cluster_ids = sorted(int(label) for label in np.unique(labels) if label >= 0)
    noise_count = int(np.sum(labels < 0))

    if n_songs == 0:
        status = "no_ovenbird_songs_detected"
    elif n_songs < min_cluster_size:
        status = "insufficient_song_events"
    elif not cluster_ids:
        status = "no_stable_clusters"
    else:
        status = "candidate_count_available"

    return {
        "status": status,
        "candidate_individual_count": len(cluster_ids),
        "detected_song_events": int(n_songs),
        "clustered_song_events": int(n_songs - noise_count),
        "unclustered_song_events": noise_count,
        "min_cluster_size": int(min_cluster_size),
        "interpretation": (
            "Count represents stable acoustic clusters with repeated songs; "
            "it is not a guaranteed census of all birds present."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate repeatedly singing, acoustically distinct Ovenbirds in one "
            "recording using the included feature extractor."
        )
    )
    parser.add_argument("audio", type=Path, help="Long WAV/MP3 recording to analyze")
    parser.add_argument(
        "--detections-csv",
        type=Path,
        help=(
            "Optional CSV with start_time/end_time and optional score columns. "
            "When supplied, skip HawkEars detection."
        ),
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.0,
        help="Minimum raw HawkEars Ovenbird score (default: 0.0)",
    )
    parser.add_argument(
        "--min-event-separation",
        type=float,
        default=4.0,
        help=(
            "Minimum seconds between retained song-window centers; prevents "
            "overlapping detector windows from duplicating one song (default: 4.0)"
        ),
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum repeated song events forming an individual cluster (default: 5)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=2024)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("ovenbird_count"),
        help="Prefix for .summary.json, .songs.csv, and .embeddings.npy outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if args.min_cluster_size < 2:
        raise ValueError("--min-cluster-size must be at least 2")

    if args.detections_csv:
        detections = load_detection_table(
            args.detections_csv,
            audio_path,
            args.detection_threshold,
            args.min_event_separation,
        )
    else:
        detections = detect_ovenbird_windows(
            audio_path,
            args.detection_threshold,
            args.min_event_separation,
            args.batch_size,
            args.num_workers,
        )

    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    if detections.empty:
        labels = np.array([], dtype=int)
        embeddings = np.empty((0, 512))
    else:
        windows = make_embedding_windows(detections, audio_path)
        embeddings = embed_song_windows(windows, args.batch_size, args.num_workers)
        labels, _ = cluster_song_embeddings(
            embeddings,
            args.min_cluster_size,
            args.random_state,
        )
        detections["candidate_individual"] = labels

    summary = summarize(labels, len(detections), args.min_cluster_size)
    summary["audio"] = str(audio_path)

    detections.to_csv(prefix.with_suffix(".songs.csv"), index=False)
    np.save(prefix.with_suffix(".embeddings.npy"), embeddings)
    prefix.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
