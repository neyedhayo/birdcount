"""Refresh the data and audio used by the local Birdcount deck."""

import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from individual_counting_experiment import (
    RANDOM_STATE,
    fit_pairwise_mlp,
    load_embedding_data,
    location_split,
    pair_features,
    pairwise_mlp_labels,
    recording_groups,
)

MEDIA = HERE / "media"
EXPERIMENT = ROOT / "outputs/individual_counting_experiment"
EMBEDDINGS = ROOT / "outputs/ovenbird_embeddings"
SAMPLE_RATE = 24_000
METHOD_LABELS = {
    "agglomerative_average": "Average linkage",
    "cosine_adjacency": "Cosine graph",
    "agglomerative_complete": "Complete linkage",
    "pairwise_mlp": "Pairwise MLP",
    "hdbscan": "HDBSCAN",
    "dpgmm": "DP-GMM",
}


def read_samples(path):
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar",
        str(SAMPLE_RATE), "-f", "f32le", "-",
    ])
    return np.frombuffer(raw, dtype="<f4")


def waveform(samples, bins):
    windows = np.array_split(samples, bins)
    scale = max(np.quantile(np.abs(samples), 0.9995), 1e-6)
    envelope = np.array([(window.min(), window.max()) for window in windows])
    return np.clip(envelope / scale, -1, 1).round(3).tolist()


def sample_audio(path, name, start, duration):
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", str(path),
        "-t", str(duration), "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-af", "loudnorm=I=-20:TP=-2:LRA=8", "-c:a", "libmp3lame",
        "-b:a", "64k", str(MEDIA / name),
    ], check=True)


def pairwise_example(metadata, embeddings, song_rows, cutoff):
    development = location_split(metadata).eq("development")
    groups = recording_groups(metadata, development)
    pca = PCA(n_components=64, random_state=RANDOM_STATE)
    pca.fit(embeddings[development])
    features = pca.transform(embeddings).astype(np.float32)
    model = fit_pairwise_mlp(metadata, features, groups, "pca_64")
    songs = features[song_rows]
    left, right = np.array(list(combinations(range(len(songs)), 2))).T
    scores = model.estimator.predict_proba(pair_features(songs[left], songs[right]))[:, 1]
    return {
        "features": songs[:, ::4].round(4).tolist(),
        "pairs": [[int(a), int(b), float(score)] for a, b, score in zip(left, right, scores)],
        "cutoff": cutoff,
        "labels": pairwise_mlp_labels(songs, model, cutoff).tolist(),
    }


def main():
    MEDIA.mkdir(exist_ok=True)
    detection_wav = ROOT / "Recordings/199823.wav"
    samples = read_samples(detection_wav)
    detections = pd.read_csv(
        ROOT / "ovenbird-individual-recognition/results/my_recording.songs.csv"
    )
    sample_audio(detection_wav, "field-recording.mp3", 0, 60)
    detected_calls = []
    for number, row in enumerate(detections.itertuples(), start=1):
        # Detections come from three-second windows. The embedder uses the
        # centered two seconds, as in count_individuals.make_embedding_windows.
        start = float(row.center_time) - 1
        path = f"detected-song-{number}.mp3"
        sample_audio(detection_wav, path, start, 2)
        clip = samples[round(start * SAMPLE_RATE):round((start + 2) * SAMPLE_RATE)]
        detected_calls.append({
            "start": start, "center": row.center_time, "audio": path,
            "wave": waveform(clip, 72),
        })

    metadata, embeddings = load_embedding_data(
        EMBEDDINGS / "ovenbird_embedding_metadata.csv", EMBEDDINGS / "ovenbird_embeddings.npy"
    )
    annotated = metadata.loc[metadata["recording_id"].eq(220875)].sort_values("song_center_time")
    song_rows = annotated["embedding_row"]
    vectors = embeddings[song_rows]
    song_distances = pdist(vectors, metric="cosine")
    annotated_wav = ROOT / "Recordings/220875.wav"
    annotated_samples = read_samples(annotated_wav)
    comparison_clips = []
    for number, row in enumerate(annotated.itertuples(), start=1):
        start, end = row.start_time, row.end_time
        clip = annotated_samples[round(start * SAMPLE_RATE):round(end * SAMPLE_RATE)]
        path = f"annotated-song-{number}.mp3"
        sample_audio(annotated_wav, path, start, end - start)
        comparison_clips.append({
            "audio": path,
            "wave": waveform(clip, 72),
            "features": vectors[number - 1, ::12].round(3).tolist(),
        })

    test = pd.read_csv(EXPERIMENT / "test_recording_results.csv").query(
        "method == 'agglomerative_average' and representation == 'embedding_512'"
    )
    counts = pd.crosstab(test["true_count"], test["predicted_count"])
    count_range = list(range(1, max(max(counts.index), max(counts.columns)) + 1))
    count_matrix = counts.reindex(index=count_range, columns=count_range, fill_value=0)
    # Match the representation selection used in the repo's held-out results table.
    inductive = pd.read_csv(EXPERIMENT / "test_summary_results.csv").query(
        "partition == 'test' and evaluation_regime == 'inductive' and method in @METHOD_LABELS"
    )
    best_representations = inductive.groupby("method")["hungarian_accuracy"].idxmax()
    ranked_methods = inductive.loc[best_representations].sort_values(
        "count_accuracy", ascending=False
    )
    mlp_settings = inductive.query("method == 'pairwise_mlp' and representation == 'pca_64'").iloc[0]
    mlp_cutoff = json.loads(mlp_settings.parameters)["probability_threshold"]
    data = {
        "fieldWave": waveform(samples, 600),
        "detectedCalls": detected_calls,
        "annotatedCalls": comparison_clips,
        "projection": PCA(n_components=2).fit_transform(vectors).round(5).tolist(),
        "songDistances": squareform(song_distances).round(6).tolist(),
        "merges": linkage(song_distances, method="average").round(6).tolist(),
        "pairwiseMLP": pairwise_example(metadata, embeddings, song_rows, mlp_cutoff),
        "countMatrix": count_matrix.to_numpy().tolist(),
        "methodAccuracy": [
            {
                "label": METHOD_LABELS[row.method],
                "accuracy": row.count_accuracy,
                "recordings": int(row.n_recordings),
            }
            for row in ranked_methods.itertuples()
        ],
    }
    contents = json.dumps(data, separators=(",", ":"))
    (HERE / "data.js").write_text(
        "// Generated by refresh-data.py from the Birdcount experiment and audio.\n"
        f"const birdcountData = {contents};\n"
    )


if __name__ == "__main__":
    main()
