"""Reusable utilities for the individual-counting embedding experiment.

The unit being clustered is a call, and clustering is performed independently
inside each recording. Locations, rather than calls, are split between the
development and test partitions to avoid spatial leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


@dataclass
class PairwiseModel:
    """A fitted pairwise classifier and its representation name."""

    estimator: object
    representation: str


def load_embedding_data(metadata_path, embedding_path):
    """Load metadata and align it exactly to the rows in the embedding array."""
    metadata = pd.read_csv(metadata_path)
    embeddings = np.load(embedding_path)
    usable = metadata.loc[
        metadata["embedding_status"].eq("embedded")
        & metadata["embedding_row"].notna()
    ].copy()
    usable["embedding_row"] = usable["embedding_row"].astype(int)
    usable = usable.sort_values("embedding_row").reset_index(drop=True)
    if not np.array_equal(usable["embedding_row"].to_numpy(), np.arange(len(embeddings))):
        raise ValueError("Metadata embedding_row values do not align with the embedding array")
    usable["individual_order"] = usable["individual_order"].astype(int)
    return usable, embeddings.astype(np.float32, copy=False)


def location_split(metadata, development_fraction=0.60, random_state=RANDOM_STATE):
    """Choose a reproducible location split close to the requested row fraction.

    Several group-level random splits are considered because locations contain
    unequal numbers of calls. Selection uses only group sizes, never labels.
    """
    locations = metadata["location"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(random_state)
    target_n_locations = round(development_fraction * len(locations))
    target_n_rows = development_fraction * len(metadata)
    best = None
    for _ in range(2_000):
        shuffled = rng.permutation(locations)
        dev_locations = frozenset(shuffled[:target_n_locations])
        n_rows = metadata["location"].isin(dev_locations).sum()
        score = abs(n_rows - target_n_rows)
        if best is None or score < best[0]:
            best = (score, dev_locations)
            if score == 0:
                break
    dev_locations = best[1]
    partition = np.where(metadata["location"].isin(dev_locations), "development", "test")
    return pd.Series(partition, index=metadata.index, name="partition")


def recording_groups(metadata, partition_mask, minimum_calls=2):
    """Return recording row indices for non-trivial groups in one partition."""
    subset = metadata.loc[partition_mask]
    return {
        recording_id: frame.index.to_numpy()
        for recording_id, frame in subset.groupby("recording_id", sort=False)
        if len(frame) >= minimum_calls
    }


def labels_from_adjacency(adjacency):
    return connected_components(adjacency, directed=False, return_labels=True)[1]


def cosine_adjacency_labels(x, threshold):
    adjacency = cosine_similarity(x) >= threshold
    return labels_from_adjacency(adjacency)


def agglomerative_labels(x, linkage, distance_threshold):
    if len(x) == 1:
        return np.zeros(1, dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage=linkage,
        distance_threshold=distance_threshold,
    )
    return model.fit_predict(x)


def hdbscan_labels(x, min_cluster_size=2, min_samples=1):
    """Cluster calls and convert each noise point (-1) to a singleton cluster."""
    if len(x) < min_cluster_size:
        return np.arange(len(x))
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        allow_single_cluster=True,
    ).fit_predict(x)
    labels = labels.astype(int, copy=True)
    next_label = labels.max(initial=-1) + 1
    for row in np.flatnonzero(labels == -1):
        labels[row] = next_label
        next_label += 1
    return labels


def dpgmm_labels(x, concentration_prior=1.0, random_state=RANDOM_STATE):
    """Fit a finite truncation of a Dirichlet-process Gaussian mixture."""
    if len(x) == 1:
        return np.zeros(1, dtype=int)
    model = BayesianGaussianMixture(
        n_components=len(x),
        covariance_type="diag",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=concentration_prior,
        max_iter=500,
        n_init=3,
        reg_covar=1e-5,
        random_state=random_state,
    )
    return model.fit_predict(x)


def pair_features(x1, x2):
    """Symmetric pair representation used by the pairwise MLP."""
    return np.concatenate([np.abs(x1 - x2), x1 * x2], axis=-1)


def make_pair_dataset(metadata, features, groups, random_state=RANDOM_STATE):
    """Create balanced positive/negative call pairs within recordings."""
    rng = np.random.default_rng(random_state)
    positive, negative = [], []
    for indices in groups.values():
        truth = metadata.loc[indices, "individual_order"].to_numpy()
        for left, right in combinations(range(len(indices)), 2):
            item = (indices[left], indices[right])
            (positive if truth[left] == truth[right] else negative).append(item)
    if not positive or not negative:
        raise ValueError("Both positive and negative training pairs are required")
    n = min(len(positive), len(negative))
    selected = positive if len(positive) == n else list(rng.choice(positive, n, replace=False))
    selected_negative = negative if len(negative) == n else list(rng.choice(negative, n, replace=False))
    pairs = selected + selected_negative
    y = np.r_[np.ones(n, dtype=int), np.zeros(n, dtype=int)]
    order = rng.permutation(len(y))
    left = np.array([pairs[i][0] for i in order])
    right = np.array([pairs[i][1] for i in order])
    return pair_features(features[left], features[right]), y[order]


def fit_pairwise_mlp(metadata, features, groups, representation, random_state=RANDOM_STATE):
    x_pairs, y_pairs = make_pair_dataset(metadata, features, groups, random_state)
    estimator = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 16),
            activation="relu",
            alpha=1e-3,
            early_stopping=True,
            validation_fraction=0.2,
            max_iter=500,
            random_state=random_state,
        ),
    )
    estimator.fit(x_pairs, y_pairs)
    return PairwiseModel(estimator=estimator, representation=representation)


def pairwise_mlp_labels(x, model, probability_threshold):
    n = len(x)
    adjacency = np.eye(n, dtype=bool)
    pairs = list(combinations(range(n), 2))
    if pairs:
        left = np.array([p[0] for p in pairs])
        right = np.array([p[1] for p in pairs])
        probability = model.estimator.predict_proba(pair_features(x[left], x[right]))[:, 1]
        for (i, j), is_same in zip(pairs, probability >= probability_threshold):
            adjacency[i, j] = adjacency[j, i] = is_same
    return labels_from_adjacency(adjacency)


def hungarian_accuracy(y_true, y_pred):
    """Permutation-invariant call-level clustering accuracy."""
    contingency = confusion_matrix(y_true, y_pred)
    rows, cols = linear_sum_assignment(-contingency)
    return contingency[rows, cols].sum() / len(y_true)


def evaluate_configuration(metadata, features, groups, label_function):
    """Evaluate one clustering configuration for every recording."""
    rows = []
    for recording_id, indices in groups.items():
        y_true = metadata.loc[indices, "individual_order"].to_numpy()
        y_pred = np.asarray(label_function(features[indices]))
        true_count = np.unique(y_true).size
        predicted_count = np.unique(y_pred).size
        rows.append(
            {
                "recording_id": recording_id,
                "n_calls": len(indices),
                "true_count": true_count,
                "predicted_count": predicted_count,
                "signed_error": predicted_count - true_count,
                "absolute_error": abs(predicted_count - true_count),
                "exact_count": predicted_count == true_count,
                "hungarian_accuracy": hungarian_accuracy(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def summarize_recording_results(recording_results):
    """Macro-average recordings so long recordings do not dominate."""
    return pd.Series(
        {
            "n_recordings": len(recording_results),
            "count_accuracy": recording_results["exact_count"].mean(),
            "mean_signed_error": recording_results["signed_error"].mean(),
            "count_mae": recording_results["absolute_error"].mean(),
            "hungarian_accuracy": recording_results["hungarian_accuracy"].mean(),
        }
    )


def configuration_score(summary):
    """Lexicographic model-selection score: assignment, count accuracy, then MAE."""
    return (
        summary["hungarian_accuracy"],
        summary["count_accuracy"],
        -summary["count_mae"],
    )
