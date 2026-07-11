# Breast Cancer Segmentation - Semi-Supervised AI System

A semi-supervised deep learning pipeline for automatic tumour segmentation in breast
DCE-MRI. An Attention U-Net is first trained on the small set of expert-annotated
patients, then used to generate confidence-filtered pseudo-labels for the large
unlabelled pool, and finally a second model is trained on the combined data.

Course project for *Advanced Data Analysis & Artificial Intelligence*, Summer Term 2026,
Master Medical & Sports Technologies, MCI Management Center Innsbruck.

Authors: Melisa Ata, Julia Hampersberger, Sabrina Staud.

---

## Overview

The pipeline runs in three notebooks, in this order:

1. **`Preprocessing.ipynb`** - inspects the raw DICOM dataset, selects the correct MRI
   series per patient, normalises and resizes the slices, aligns the ROI masks, and
   verifies the result.
2. **`Semi_Supervised_AI_System.ipynb`** - writes the chunked dataset, runs the
   exploratory data analysis and integrity checks, builds the train/validation/test
   split, trains the supervised Attention U-Net baseline, generates pseudo-labels for
   the unlabelled patients, and trains the semi-supervised model.
3. **`Evaluation.ipynb`** - evaluates both models on the held-out test patients and
   produces the comparison plots and error maps.

---

## Dataset

The project uses the **Advanced-MRI-Breast-Lesions** collection, a single-institution
DCE-MRI dataset distributed through The Cancer Imaging Archive (TCIA).

- DOI: [10.7937/C7X1-YN57](https://doi.org/10.7937/C7X1-YN57)
- 632 patients in total, of which 99 include expert ROI tumour masks.
- The raw collection is roughly 600 GB and is kept on an external SSD.

After preprocessing, 613 patients remain (94 labelled, 519 unlabelled), stored as
chunked pickle files of about 13.6 GB.

The raw dataset is **not** included in this repository. Download it from TCIA and point
the notebook paths to your local copy (see *Setup* below).

---

## Repository structure

```
AI_PROJECT/
├── Preprocessing.ipynb              # step 1: DICOM inspection and preprocessing
├── Semi_Supervised_AI_System.ipynb  # step 2: chunking, EDA, training, pseudo-labels
├── Evaluation.ipynb                 # step 3: evaluation and comparison
├── Preprocessing_data/              # artefacts written during preprocessing
│   ├── dataset_survey.csv               # per-folder survey of every patient series
│   ├── patient_folder_overview.png      # per-folder slice-preview grid
│   └── patients_without_registered_ax.txt  # patients with no registered AX Vibrant series
├── Processed_and_Split_data/
│   ├── manifest.csv                     # one row per patient (chunk, has_roi, n_slices, status)
│   └── split.json                       # version-controlled train/val/test patient IDs
├── results/
│   └── models/
│       ├── attn_unet_supervised.keras   # supervised baseline
│       ├── attn_unet_semisup.keras      # semi-supervised model
│       ├── history_supervised.csv       # per-epoch training log (baseline)
│       └── history_semisup.csv          # per-epoch training log (semi-supervised)
├── Advanced-MRI-Breast-Lesions-DA-Clinical-Sep2024.xlsx  # clinical metadata (from TCIA)
├── Project_Proposal_...pdf          # project proposal
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

The large data lives outside the repository, on the external SSD:

```
<SSD>/Advanced-MRI-Breast-Lesions/
├── DICOM Images/manifest-.../Advanced-MRI-Breast-Lesions/   # raw DICOM (~600 GB)
└── data/
    ├── chunks_semisup/   chunk_*.pkl   # preprocessed dataset (~13.6 GB)
    └── chunks_pseudo/    chunk_*.pkl   # generated pseudo-labels
```

---

## Requirements

- Python 3.11.11
- TensorFlow 2.17
- NumPy, pandas, pydicom, matplotlib
- OpenCV (`opencv-python`); SciPy is used as a fallback if OpenCV is missing

Install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The core packages are TensorFlow, NumPy, pandas, pydicom, matplotlib and OpenCV
(`opencv-python`); SciPy is used as a fallback if OpenCV is missing.

A GPU is recommended but not required. The models were trained on a 6 GB GPU, which is
why a reduced base width (32) and mixed-precision training are used. Training also runs
on the CPU, only slower.

---

## Setup

The data paths are defined at the top of the relevant cells and need to be adjusted to
your machine. The main ones are:

- `DICOM_FOLDER` - the raw DICOM root, e.g.
  `.../manifest-1713182663002/Advanced-MRI-Breast-Lesions`
- `OUT_DIR` / `CHUNK_DIR` - where the preprocessed `chunk_*.pkl` files are written and read
- `PSEUDO_DIR` - where the pseudo-label chunks are written and read

`split.json` and `manifest.csv` are located automatically under the working directory,
so no path change is needed for those as long as the repository is the working directory.

---

## Usage

Run the three notebooks in order.

### 1. Preprocessing

`Preprocessing.ipynb` scans the dataset, selects the *Registered AX Sen Vibrant C* series
per patient, and verifies the series selection and ROI overlay on a sample patient. This
notebook is mainly for inspection and quality control.

### 2. Chunking, training and pseudo-labelling

`Semi_Supervised_AI_System.ipynb`:

- writes the chunked dataset (`chunk_*.pkl` + `manifest.csv`),
- runs the integrity check and EDA plots,
- builds the patient-level split (`split.json`, fixed seed 42, 70/15/15 over labelled
  patients; all unlabelled patients go to the training pool),
- trains the supervised baseline and saves it to `results/models/attn_unet_supervised.keras`,
- generates confidence-filtered pseudo-labels into `chunks_pseudo/`,
- trains the semi-supervised model and saves it to `results/models/attn_unet_semisup.keras`.

### 3. Evaluation

`Evaluation.ipynb` loads both saved models and reports per-patient Dice and IoU on the 14
held-out test patients, plus Dice distributions, probability heatmaps, error maps, and a
per-patient baseline-versus-semi-supervised comparison.

---

## Model and training

- **Architecture:** Attention U-Net (four encoder/decoder levels, attention gates on every
  skip connection), base width 32, single-channel `256 x 256` input, sigmoid output.
- **Loss:** Focal Tversky (alpha = 0.9, beta = 0.1, gamma = 1.0), which penalises missed
  tumour pixels harder than false positives.
- **Class imbalance:** lesion oversampling (about 50% tumour slices per training batch)
  and synchronised flips/rotations on image and mask.
- **Optimiser:** Adam, learning rate 1e-4, batch size 8, up to 60 epochs with early
  stopping (patience 12) and learning-rate reduction on plateau.
- **Semi-supervised batches:** 50% real lesion slices, 25% pseudo-labelled lesion slices,
  25% real background slices, so real annotations stay the dominant signal.

---

## Results

Evaluated on 14 held-out test patients, tumour-containing slices only:

| Model               | Mean Dice | Mean IoU |
|---------------------|-----------|----------|
| Supervised baseline | 0.232     | 0.156    |
| Semi-supervised     | 0.253     | 0.174    |

The semi-supervised model improves the mean Dice by 0.021. The gain is largest on cases
where the baseline had failed entirely, and slightly negative on a few cases the baseline
had already solved well. The scores are modest, which is consistent with the very small
lesions (over half of the annotated slices contain fewer than 100 tumour pixels) and the
limited number of labelled training patients.

---

## Notes and limitations

- The model is fully 2D and processes each slice independently, so inter-slice context is
  not used.
- The test set is small (14 patients), so the mean scores carry high variance.
- Pseudo-labels are generated once from the baseline and kept fixed during semi-supervised
  training.

See the report for a fuller discussion and possible extensions (3D/2.5D architecture,
Mean-Teacher pseudo-labelling, multi-phase input, test-time augmentation).

---

## Data citation

Daniels, D., Last, D., Cohen, K., Mardor, Y., Sklair-Levy, M. (2024). *Standard and Delayed
Contrast-Enhanced MRI of Malignant and Benign Breast Lesions with Histological and Clinical
Supporting Data (Advanced-MRI-Breast-Lesions).* The Cancer Imaging Archive.
https://doi.org/10.7937/C7X1-YN57
