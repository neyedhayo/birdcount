# Ovenbird Individual Recognition: Detailed Repository Guide

This guide explains what is in this repository, how the pieces fit together, and how to run the main workflows. It is a companion to [README.md](README.md), not a replacement for the paper or the code itself.

## 1. What this project does

The project identifies individual Ovenbirds from their songs and uses those identifications for ecological analysis. The central workflow is:

1. Detect Ovenbird songs in long passive acoustic monitoring (PAM) recordings.
2. Turn each song into a fixed-length spectrogram.
3. Pass the spectrogram through a convolutional neural network to obtain an acoustic embedding.
4. Reduce and cluster the embeddings so that songs with similar individual characteristics receive the same candidate identity.
5. Review candidate clusters and convert them into yearly detection histories.
6. Use those histories in abundance and Cormack–Jolly–Seber (CJS) survival models.

The key training idea is to use recording point or localization array as a proxy label. Birds recorded at different points are assumed to be different often enough for a model to learn identity-relevant song features without requiring individual labels for the entire training set.

## 2. What can be run from the repository as downloaded?

| Workflow | Included inputs | Expected status |
|---|---|---|
| Demo on 100 labeled songs | Sample MP3s, labels, checkpoint | The best starting point; designed to run locally |
| Core embedding and clustering functions | Source modules and checkpoint | Reusable after adapting a clip table |
| Inspect existing results and figures | CSV results and PDF figures | Fully available without rerunning models |
| Ecological models | Resource CSVs, R scripts, JAGS model files | Mostly reproducible after installing R/JAGS dependencies |
| Evaluation on the full labeled set | Evaluation code only | Requires the external localization dataset and some path edits |
| Full PAM resighting pipeline | Code plus some derived resources | Requires the external PAM dataset and manual review steps |
| Build the localization dataset from raw audio | Code only | Reference workflow; raw audio is not included and several paths are redacted |
| Retrain the feature extractor | Training code and configs | Requires training clips, validation clips, path edits, compute, and Weights & Biases access or a small code change |

The numbered folders are best understood as the analysis history behind the manuscript. They are not a single command-line application, and several scripts assume a particular working directory, external dataset layout, or HPC filesystem.

## 3. Repository map

```text
ovenbird-individual-recognition/
├── README.md                    Short project overview
├── DETAILED_GUIDE.md            This guide
├── demo.ipynb                   Self-contained model demonstration
├── requirements.txt             Main Python dependencies, without pinned versions
├── conda_env.yml                Small historical Linux environment export
├── conda_env_april2026.yml      Large, newer macOS/Apple Silicon environment export
├── LICENSE
├── checkpoints/
│   └── full_..._best.pth        Trained ResNet-18 state dictionary
├── sample_data/
│   ├── labeled_clips_sample.csv Metadata for the demonstration
│   └── audio_clips/             100 MP3 clips from 10 individuals
├── src/                         Reusable Python implementation
├── scripts/                     Numbered manuscript workflows
│   ├── 1_create_evaluation_sets_from_localization/
│   ├── 2_train_feature_extractors/
│   ├── 3_evaluate_individual_id/
│   ├── 4_PAM_resighting/
│   ├── 5_ecological_modeling/
│   └── resources/               Small derived CSV inputs
├── results/                     Saved tabular results
└── figures/                     Saved manuscript and supplementary figures
```

## 4. Environment setup

### Recommended local setup

The Bioacoustics Model Zoo version used by the repository requires Python 3.10–3.13.
Python 3.14 is not compatible. From the repository root, create a project-local
Python 3.11 environment with `uv`:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-inference.txt
uv pip install --python .venv/bin/python jupyterlab
source .venv/bin/activate
python --version  # should report Python 3.11.x
```

Then verify the most important imports:

```bash
python -c "import torch, opensoundscape, sklearn, umap, hdbscan; print('imports OK')"
```

Notes:

- `requirements.txt` is the clearest general-purpose dependency list, but it does not pin versions. Save a lock file or environment export if exact reproducibility matters.
- `conda_env.yml` is a historical, minimal Linux export and does not contain the complete model stack.
- `conda_env_april2026.yml` is much more complete but is platform-specific and contains many packages unrelated to the core workflow.
- PyTorch installation can vary by operating system and accelerator. A CUDA system may need the wheel recommended for its CUDA version.
- Some scripts use packages not listed in `requirements.txt`, including `joblib`, `librosa`, `pytz`, `aru-metadata-parser`, and notebook tooling.
- The model downloads ImageNet ResNet weights when it is constructed. The first run therefore normally needs internet access unless those weights are already cached. The included Ovenbird checkpoint alone does not prevent that initial torchvision download.

### Working-directory rule

Relative paths are common throughout the repository:

- Run `demo.ipynb` with the repository root as the notebook working directory.
- Run training from `scripts/2_train_feature_extractors/`.
- Run evaluation scripts from `scripts/3_evaluate_individual_id/`.
- Run PAM scripts from `scripts/4_PAM_resighting/`.
- Run R scripts from `scripts/5_ecological_modeling/`.

Running a script from another directory can break both `sys.path.append("../../src/")` and its data paths.

## 5. Quick start: run the included demo

The demo embeds 100 clips, reduces the 512-dimensional embeddings to two dimensions with t-SNE, plots them by known individual, and provides audio and spectrogram examples.

1. Activate the environment.
2. Change to the repository root.
3. Start Jupyter.
4. Open and run `demo.ipynb` from top to bottom.

```bash
conda activate ovenbirds
cd ovenbird-individual-recognition
jupyter lab demo.ipynb
```

The important demo calls are:

```python
audio_clip_table = pd.read_csv(
    "./sample_data/labeled_clips_sample.csv"
).set_index(["file", "start_time", "end_time"])

cnn = load_ovenbird_model()
embeddings = cnn.embed(audio_clip_table)

embeddings_2d = evaluation.reduce_dims(
    embeddings.values,
    reduction_algorithm="tsne",
    reduced_n_dimensions=2,
)
```

Expected data and output:

- Input table: 100 rows, one per audio clip.
- Model input: two-second, 2–10 kHz spectrograms at 32 kHz sample rate.
- Embedding output: one 512-value vector per row.
- Plot: one point per song, colored by the known individual ID.

If the checkpoint cannot be found, confirm that this file exists:

```text
checkpoints/full_2025-04-10T11:02:36.028451_best.pth
```

If sample audio cannot be found, confirm that Jupyter's current directory is the repository root rather than the directory above it.

## 6. Use the trained model on your own clips

### Estimate candidate individuals in one long recording

You do not need to retrain the model. Use the repository's inference script from the
repository root:

```bash
# One-time environment setup:
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-inference.txt
source .venv/bin/activate

python count_individuals.py /absolute/path/to/recording.wav \
  --output-prefix results/my_recording
```

The script performs four steps:

1. HawkEars scores overlapping three-second windows for Ovenbird presence.
2. Overlapping detections of the same likely song event are collapsed.
3. The included checkpoint creates a 512-dimensional embedding for each retained song.
4. UMAP and HDBSCAN group repeated songs into candidate individuals.

Outputs are:

- `results/my_recording.summary.json`: count, status, and diagnostics.
- `results/my_recording.songs.csv`: each detected song and its candidate cluster.
- `results/my_recording.embeddings.npy`: original embeddings for later re-analysis.

The `candidate_individual_count` is the number of stable non-noise clusters. A
`candidate_individual` value of `-1` means the song was not assigned to a stable
cluster. If the status is `insufficient_song_events` or `no_stable_clusters`, do not
interpret a count of zero as evidence that no Ovenbirds were present.

By default, retained song centers must be at least four seconds apart. This prevents
several overlapping HawkEars windows around one song from being mistaken for several
independent songs. Change `--min-event-separation` only after inspecting the per-song
CSV and listening to the proposed events.

This approach needs a long enough recording for each vocal individual to sing several
times. With the default `--min-cluster-size 5`, a bird needs approximately five
usable song events before it can form a cluster. A single two- or three-second clip
can establish that a song resembles Ovenbird, but cannot reliably distinguish how
many individuals are present.

If you already have manually reviewed song times or detections from another program,
skip HawkEars by supplying a CSV:

```csv
start_time,end_time,score
12.0,15.0,1.8
28.0,31.0,2.4
```

```bash
python count_individuals.py recording.wav \
  --detections-csv reviewed_song_times.csv \
  --output-prefix results/reviewed
```

Inspect representative audio from every resulting cluster before treating it as a
biological individual. Simultaneous overlapping singers, very sparse singing,
movement, repertoire changes, and background interference can cause under- or
over-counting.

### Required clip table

OpenSoundscape expects a DataFrame indexed by these three fields:

| Field | Meaning |
|---|---|
| `file` | Path to an audio file |
| `start_time` | Clip start in seconds |
| `end_time` | Clip end in seconds |

The model was designed for two-second Ovenbird song windows. A minimal example is:

```python
import sys
from pathlib import Path
import pandas as pd

project_root = Path("/absolute/path/to/ovenbird-individual-recognition")
sys.path.insert(0, str(project_root / "src"))

from model import load_ovenbird_model

clips = pd.DataFrame(
    {
        "file": ["/absolute/path/to/song.wav"],
        "start_time": [4.0],
        "end_time": [6.0],
    }
).set_index(["file", "start_time", "end_time"])

model = load_ovenbird_model()
embeddings = model.embed(clips)
embeddings.to_csv("ovenbird_embeddings.csv")
```

For best comparability with the included model:

- Center the window on one Ovenbird song.
- Avoid windows containing overlapping species or multiple Ovenbirds where possible.
- Keep the same preprocessing: 32 kHz resampling and a 2–10 kHz bandpass.
- Preserve row order when joining embeddings back to metadata.

The 234 outputs of the checkpoint's classification head are training-location classes, not universal individual identities. For a new field dataset, use the 512-dimensional embeddings and clustering; do not interpret the class logits as bird IDs.

### Reduce and cluster embeddings

`src/evaluation.py` provides two related routes:

- `reduce_dims(...)`: only UMAP/t-SNE reduction.
- `make_pseudolabels(...)`: preprocessing, embedding, reduction, and HDBSCAN clustering in one call.

Example using a metadata table with `song_center_time`:

```python
from model import Resnet18_Classifier
from preprocessor import OvenbirdPreprocessor
from evaluation import make_pseudolabels
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = Resnet18_Classifier(num_classes=234)
state = torch.load(project_root / "checkpoints/full_2025-04-10T11:02:36.028451_best.pth",
                   map_location=device)
model.load_state_dict(state)
model.device = device
model.to(device)

preprocessor = OvenbirdPreprocessor()

labels, reduced_features, embeddings = make_pseudolabels(
    clip_metadata,
    model=model,
    preprocessor=preprocessor,
    cluster_grouping_col="point_code",  # omit to cluster everything together
    reduction_algorithm="umap",
    reduced_n_dimensions=5,
    min_cluster_size=7,
    random_state=2024,
)
```

`clip_metadata` must have at least `file` and `song_center_time`. If `cluster_grouping_col="point_code"`, it must also contain `point_code`. A predicted label of `-1` means HDBSCAN treated that song as noise rather than assigning it to a cluster.

Practical advice:

- Cluster within recording point when the same individual is unlikely to occupy multiple points. This greatly narrows the comparison set.
- Tune `min_cluster_size` against a labeled validation set; it changes both the number of discovered individuals and the fraction left as noise.
- Fix random seeds and save the fitted/reduced output. UMAP and t-SNE can vary between runs.
- Treat clusters as candidates requiring review when they drive ecological conclusions.

## 7. Core source files

### `src/preprocessor.py`

Defines the audio-to-spectrogram pipeline.

- `OvenbirdPreprocessor`: two-second clips, 32 kHz resampling, 2–10 kHz bandpass, normalization, and optional background overlay.
- `JitterClipTime`: training augmentation that shifts the requested clip window by up to 0.5 seconds.
- `noise_and_mute` and `mute_and_normalize`: helper transformations used in experiments.

Augmentations should be bypassed for evaluation and production embedding so the same clip produces a stable input.

### `src/model.py`

Defines model architectures and checkpoint loading.

- `Resnet18_Classifier` and `Resnet50_Classifier`: embedding backbone plus location-classification head.
- `Resnet18_Embedder` and `Resnet50_Embedder`: embedding-only wrappers used by metric learning.
- `ContrastiveResnet18` and `ContrastiveResnet50`: backbone plus projection head.
- `HawkEarsOneModel`: transfer learning from one HawkEars ensemble member.
- `load_ovenbird_model`: loads the included ResNet-18 checkpoint and wraps its embedder in an OpenSoundscape `CNN` for convenient `.embed(...)` calls.

The included checkpoint must be loaded into `Resnet18_Classifier(num_classes=234)` because the saved state dictionary includes a 234-class head.

### `src/dataset.py`

Connects metadata tables to OpenSoundscape preprocessing and PyTorch data loading.

- `AIIDLocalizedClipDataset`: uses `song_center_time` and optional individual/pseudo-label columns.
- `GridAsLabelDataset`: treats localization array as the class label.
- `PointCodeDataset`: attaches `point_code`, `clip_id`, and optional pseudo-label metadata to each processed sample.
- `TrainingEventSampler`: keeps all clips from a localization event together.
- `MultipleClipVersionsSampler`: repeats a clip index so separate augmentations can be compared.
- `PointCodeSampler`: builds contrastive batches from multiple recording points.

### `src/loss.py`

`ssl_location_loss` combines four possible incentives:

1. Two augmented versions of the same clip should be similar.
2. Clips from different points should be dissimilar.
3. Clips from the same point should be similar.
4. Clips with the same non-noise pseudo-label should be similar.

The weights in a training config control the contribution of each term.

### `src/evaluation.py`

Contains the reusable evaluation pipeline:

- `embed`: batch model inference.
- `reduce_dims`: standardization plus UMAP or t-SNE.
- `cluster` and `cluster_points`: HDBSCAN clustering, either reducing within groups or reducing globally before grouped clustering.
- `make_pseudolabels`: end-to-end embedding and clustering.
- `evaluate`: ARI, NMI, FMI, homogeneity, completeness, V-measure, purity, and Hungarian-matched accuracy.
- `run_eval`: loads a `labeled_clips.csv` split and evaluates a model.
- `mean_range_table`: formats repeated-run summaries.

Hungarian-matched accuracy finds the best mapping between arbitrary cluster numbers and known individual numbers. It is a clustering score, not ordinary fixed-label classification accuracy.

### Plotting and review helpers

- `src/visualization.py` contains interactive notebook tools for annotation, sample inspection, feature exploration, and histograms.
- `src/plot_utils.py` contains common Matplotlib styling.
- `src/palette.py` defines project colors.
- `src/Gill Sans Light Regular.otf` is a bundled font used by some figures.

## 8. Full numbered workflow

### Stage 1 — Create labeled evaluation sets from localization

Folder: `scripts/1_create_evaluation_sets_from_localization/`

This stage creates a trusted individual-labeled dataset from synchronized recorder arrays. The raw 1.7 TB audio is not included, and several paths are deliberately `REDACTED`.

| Order | File | Purpose |
|---:|---|---|
| 1 | `1_detect_species_HawkEars.py` | Run HawkEars over localization-array recordings |
| 2 | `2_select_detections_to_localize.ipynb` | Aggregate scores and select candidate Ovenbird detections |
| 3 | `3_localize_ovenbird.py` | Localize songs using synchronized arrays and recorder coordinates |
| 4 | `4_automated_minspec_review.py` | Create minimum spectrogram audio and filter poor localizations |
| 5 | `5_remove_duplicate_events.py` | Remove spatial duplicates at the same timestamp with DBSCAN |
| 6 | `6_select_clips_to_annotate.ipynb` | Sample localized events for manual review |
| 7 | `7_annotate_localized_songs.ipynb` | Assign human individual labels |
| 8 | `8_resolve_annotations.ipynb` | Resolve reviewer labels by majority voting |
| 9–13 | Remaining notebooks | Produce figures and assess reviewer, species-classifier, and spatial-label agreement |

To adapt this stage:

1. Supply synchronized recordings and a recorder-coordinate table.
2. Replace every `REDACTED` path.
3. Confirm timestamp parsing and timezone assumptions in `get_dt`; the script uses `US/Mountain`.
4. Inspect the HawkEars class name and thresholds for the model version you install.
5. Run each step interactively and verify its saved table before continuing.

This stage is not needed if you already have a table of labeled, centered Ovenbird songs.

### Stage 2 — Train feature extractors

Folder: `scripts/2_train_feature_extractors/`

The main command is:

```bash
cd scripts/2_train_feature_extractors
python train.py train_configs/my_config.yml
```

Do not use `base.yml` unchanged. Copy it and edit the four paths:

```yaml
paths:
  save_dir: /absolute/path/to/training_results
  train_clips_path: /absolute/path/to/train_clips
  background_clips_path: /absolute/path/to/background_clips
  labeled_clips_path: /absolute/path/to/labeled_clips
```

Expected training data:

```text
train_clips_path/
├── ovenbird_train_clips.csv
└── <audio files named by clip_id>
```

`ovenbird_train_clips.csv` needs at least:

- `clip_id`: audio filename relative to `train_clips_path`.
- `point_code`: recording-location proxy label.

The script assumes each source clip is five seconds long and centers the configured training duration within it.

For overlay augmentation, `background_clips_path/background_clips.csv` must contain `clip_id` values pointing to background audio files. For per-epoch validation, `labeled_clips_path/labeled_clips.csv` must contain `rel_path`, `data_split`, and `aiid_label`.

Training strategies are selected by `training.loss_fn`:

| Setting | Strategy |
|---|---|
| `cross_entropy_loss` | Multiclass location classification |
| `binary_cross_entropy_loss` | One-hot location classification |
| `subcenter_arcface_loss` | ArcFace metric learning |
| `contrastive_point_loss` | Point/augmentation contrastive learning |
| `ssl_contrastive_point_loss` | Intended iterative pseudo-label contrastive learning; see caveats below |

Each repeat creates a timestamped output directory containing `metrics.csv`, `config.yml`, and `loss.csv`. A `best.pth` checkpoint is written only when `save_checkpoints: true` and validation ARI improves.

Other files:

- `1_create_train_experiments.py` expands `base.yml` into a grid of experiment configs and Slurm jobs.
- `2_launch_slurm_jobs.sh` submits generated jobs.
- `slurm_scripts/` contains cluster-specific launch scripts.
- `3_summarize_training_experiments.ipynb` combines experiment outputs.
- `train_configs/` records the manuscript's configurations.

### Stage 3 — Evaluate individual identification

Folder: `scripts/3_evaluate_individual_id/`

The evaluation workflow compares the custom feature extractor with BirdNET and HawkEars embeddings, tunes dimensionality reduction on validation data, evaluates once on test data, and creates summary figures.

Recommended sequence:

1. Download or prepare `localization_dataset_labeled_clips/labeled_clips.csv` plus its audio.
2. Edit `labels_dir` and checkpoint paths in the scripts/notebooks.
3. Generate BirdNET embeddings with `1_generate_birdnet_embeddings.ipynb` if that baseline is needed.
4. Generate HawkEars embeddings with `2_generate_hawkears_embeddings.ipynb`.
5. Run `3_evaluate_dim_reduction_validation_set.py` to select UMAP/t-SNE settings.
6. Run `4_evaluate_dim_reduction_test_set.py` once using the chosen settings.
7. Use notebooks 5–10 for plots, baseline comparisons, cross-point contamination, and closed-set experiments.

The labeled table is expected to contain:

- `rel_path`: audio path relative to the label directory.
- `data_split`: `val` or `test`.
- `aiid_label`: integer individual identity, with optional `no_song` rows filtered out.
- Songs centered at second 5 in ten-second source clips.

Saved summary tables are already available in `results/validation_set_performance*.csv`, `results/test_set_performance*.csv`, and related experiment CSVs.

### Stage 4 — Discover and resight individuals in PAM data

Folder: `scripts/4_PAM_resighting/`

This is the longest workflow and mixes automation with manual review.

| Order | File | Purpose |
|---:|---|---|
| 1 | `1_detect_Ovenbird_hawkears.py` | Run HawkEars over PAM recordings |
| 2 | `2_select_dets_no_other_sp.py` | Retain high-quality Ovenbird candidates and remove likely overlaps |
| 3 | `3_detect_centered_Ovenbird_songs.py` | Refine song center times |
| 4 | `4_subset_detections_by_date.ipynb` | Restrict to comparable dates, times, and monitored points |
| 5 | `5_extract_pam_dataset_clips.py` | Export three-second clips and create the public clip dataset |
| 6 | `6_select_random_clips_for_review.py` | Draw a random validation sample for species review |
| 7 | `7_embed_and_reduce_dims.py` | Generate 512-D embeddings and save standardized 3-D t-SNE features |
| 8 | `8_inspect_cluster_parameter_FigureS3.ipynb` | Inspect HDBSCAN sensitivity |
| 9 | `9_cluster_features.py` | Cluster songs separately within each point |
| 10 | `10_review_phase1_clusters.ipynb` | Review cluster coherence |
| 11 | `11_update_clusters_after_review.ipynb` | Apply merge/split/reject decisions |
| 12 | `12_review_phase2_detection_hist.ipynb` | Review candidate identities through time |
| 13–16 | Remaining notebooks | Explore histories, across-year songs, cross-point matches, and spatial autocorrelation |

The reusable public-dataset layout expected by steps 5 onward is:

```text
pam_dataset_v4/
├── pam_dataset_clips.csv
├── audio/
│   └── <clip_name files>
├── oven_clips_aiid_embeddings_tsne-3d.npy
├── per_point_clusters.csv
└── per_point_clusters_sample10.csv
```

`pam_dataset_clips.csv` needs at least `clip_name` and `point_code`. Earlier extraction also needs the original `file`, `start_time`, and other temporal/site metadata.

Step 7 saves only the standardized 3-D t-SNE representation, not the original 512-D embeddings. If future analysis may need a different reduction method, save `embeddings` as a second `.npy` file before t-SNE.

Step 9 evaluates several HDBSCAN minimum sample sizes and uses the `cluster_30` solution for its review sample and summary. Cluster labels are prefixed with `point_code` so numeric cluster IDs from different points do not collide.

### Stage 5 — Fit ecological models

Folder: `scripts/5_ecological_modeling/`

Run these scripts from their own directory because their paths are relative.

#### Survival models

`1_nimble_CJS.R` fits three CJS variants to both all identified birds and a residents-only subset:

- Full: year/block terms, site random effect, and habitat covariates.
- Null: year/block terms and site random effect without habitat covariates.
- Base: simpler constant structure.

Inputs include:

- `scripts/resources/nimble_detection_histories_and_covariates.csv`
- `scripts/resources/nimble_detection_histories_and_covariates_residents.csv`
- `scripts/resources/nimble_site_habitat_covariates.csv`
- The three `cjs_*.txt` JAGS model definitions in the same folder.

Install JAGS separately, then in R install the packages used by the script:

```r
install.packages(c(
  "AHMbook", "jagsUI", "patchwork", "ggplot2",
  "dplyr", "rjags", "tidyr"
))
```

Run:

```r
setwd("scripts/5_ecological_modeling")
source("1_nimble_CJS.R")
```

The production MCMC settings are substantial: three chains, 60,000 iterations, 30,000 burn-in, thinning by 30, and 5,000 adaptation iterations. Use the commented shorter settings only as a smoke test, not for final inference. Check convergence diagnostics and effective sample sizes before interpreting posterior summaries.

Outputs go to `results/survival_model_parameter_estimates/` and `figures/parameter_posteriors/`.

#### Abundance model

`2_glm_abundance.R` fits Poisson GLMs with site block, year, slope, canopy, and basal area, compares model subsets, and saves partial-effect plots.

Input: `scripts/resources/per_point_counts_and_covars.csv`.

Additional R packages are `ggeffects` and `MuMIn`.

#### Random-review comparison

`3_random_vs_aiid_plots.ipynb` simulates how many individuals would be found by reviewing random detections rather than using AIID. `4_fit_cjs_simulated_random_sampling.R` fits the base CJS model to those simulated histories.

## 9. Results, figures, and resource files

### `results/`

The root CSVs contain validation/test performance, training experiment summaries, dimensionality-reduction sweeps, closed-set results, and point-count analyses.

Subfolders contain:

- `abundance_model_parameter_estimates/`: GLM coefficients and model comparison.
- `survival_model_parameter_estimates/`: CJS posterior summaries and random-review simulations.
- `simultaneous_point_clustering/`: how clustering changes as more points are considered together.
- `detection_histories/`: a destination for reviewed and simulated histories; the checked-in folder is mostly empty.

### `figures/`

Contains manuscript-ready PDFs and a small number of PNGs. These are outputs, not inputs. Filenames usually describe the analysis more reliably than their figure number alone.

### `scripts/resources/`

Contains small, reusable inputs such as resolved labels, detection histories, habitat covariates, point coordinates, and a ZIP of precomputed embeddings. Before regenerating a result, inspect its notebook or script to determine whether it reads one of these resources or an external dataset.

## 10. Important data contracts

### Demonstration clip table

Required: `file`, `start_time`, `end_time`.

Useful metadata: `aiid_label`, `array`, `localization_event_id`, coordinates, and split.

### `AIIDLocalizedClipDataset` table

Required: `file`, `song_center_time`.

Optional: integer `aiid_label`, `array`, `event_id`, `pseudo_label`.

### `PointCodeDataset` table

Required: a `(file, start_time, end_time)` index and a `point_code` column.

Optional: `pseudo_label`.

### Full labeled evaluation table

Required by `run_eval`: `rel_path`, `data_split`, `aiid_label`. The code assumes the song is centered at second 5.

### Training table

Required by `train.py`: `clip_id`, `point_code`. Files are assumed to be five-second clips in `train_clips_path`.

## 11. Known caveats in the current code

These are important when moving beyond the demo:

1. Many scripts have absolute HPC paths, `REDACTED` paths, or sibling-directory assumptions. Search before running:

   ```bash
   rg -n 'REDACTED|/jet/|/ocean/|\.\./\.\./\.\./' scripts
   ```

2. The short README contains a conceptual structure, but some names do not match the checkout. For example, the actual folders are `1_create_evaluation_sets_from_localization` and `3_evaluate_individual_id`; there is no `6_results` script folder.

3. The SSL loss name is inconsistent in `train.py`: configuration selection accepts `ssl_contrastive_point_loss`, while later pseudo-label branches check `ssl_contrastive_point`. As written, the intended pseudo-label update and label use will not execute for the supplied SSL configs. Standard supervised, ArcFace, and non-SSL contrastive paths are separate from this issue.

4. `train.py` initializes a Weights & Biases run unconditionally. For offline training, configure W&B offline mode or edit the script to make logging optional.

5. Training device selection uses the configured CUDA device only when CUDA is available and otherwise falls back to CPU; it does not select Apple MPS.

6. Several output directories are created without `parents=True`. Create their parent directories before a run.

7. `scripts/4_PAM_resighting/6_select_random_clips_for_review.py` contains redacted dictionaries and an apparent malformed string (`dir = "REDACTED""`). Treat it as reference code until repaired and configured.

8. `scripts/5_ecological_modeling/4_fit_cjs_simulated_random_sampling.R` references a relative `det_hists_with_random_sampling/full_det_hist.csv` path that differs from `det_hist_dir`, and writes with an `out_dir` variable that is not defined in that script. Set those paths before running it.

9. t-SNE is useful for the manuscript workflow but does not provide a reusable transform for unseen samples. For an operational system that receives new recordings over time, retain the original embeddings and consider a fitted reducer or direct embedding-space method.

10. The pipeline discovers acoustic clusters, not guaranteed biological identities. Overlapping birds, repertoire variation, recording changes, movement between points, and sparse sampling all require ecological and manual validation.

## 12. Reproducibility checklist

Before a major run:

- Record the Git commit and keep a copy of the edited config.
- Use absolute data paths in copied configs.
- Record Python/R, OpenSoundscape, PyTorch, scikit-learn, HawkEars, and JAGS versions.
- Set NumPy, Python, PyTorch, UMAP, and t-SNE seeds where supported.
- Preserve the exact row order connecting metadata, embeddings, reduced features, and cluster labels.
- Save original 512-D embeddings as well as reduced features.
- Validate dimensionality-reduction and clustering hyperparameters only on validation data.
- Keep test data untouched until the final evaluation.
- Inspect audio from every ecological cluster or use a documented sampling protocol.
- Check CJS chain convergence before reporting posterior estimates.

## 13. Which entry point should I use?

| Goal | Start here |
|---|---|
| Understand the model visually | `demo.ipynb` |
| Embed a few new centered songs | `src/model.py::load_ovenbird_model` |
| Cluster a labeled or unlabeled clip table | `src/evaluation.py::make_pseudolabels` |
| Change audio preprocessing | `src/preprocessor.py` |
| Train a new location-supervised model | Copy `train_configs/base.yml`, then run `train.py` |
| Compare embeddings with known identities | `src/evaluation.py::run_eval` and stage 3 |
| Reproduce the PAM case study | Stage 4, with the external PAM dataset |
| Refit survival or abundance models | Stage 5 using `scripts/resources/` |
| Inspect published outputs without recomputing | `results/` and `figures/` |

For a new species or study area, begin with the demo and a small labeled validation set. Confirm that embeddings separate known individuals before investing in full detection, clustering, review, and ecological modeling.
