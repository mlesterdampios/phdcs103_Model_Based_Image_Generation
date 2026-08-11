# Group D — Common VAE Experimental Baseline

This repository contains the **shared, minimum experimental baseline** for Group D's PhD CS Image Processing activity on **Variational Autoencoder (VAE) image generation**.

The baseline deliberately covers only the common pipeline required by the activity:

1. Load and describe an image dataset.
2. Train or reload a simple VAE.
3. Reconstruct real images.
4. Generate synthetic images from random latent vectors.
5. Interpolate between two encoded images.
6. Transform images to the frequency domain using a 2D FFT.
7. Produce minimal spatial-domain, frequency-domain, and quality comparisons.
8. Save reusable arrays, metrics, and the model checkpoint.

It is **not intended to be submitted unchanged**. Each member should create an individual branch or notebook copy and expand the analysis for the assigned role.

## Role mapping

| Member | Required role | Individual focus |
|---|---|---|
| **Manasan** | Model Development / Latent-Space Behavior | Changes in image structure during interpolation |
| **Almazan** | Frequency-Domain Analysis | FFT behavior, spectral smoothness, and frequency consistency |
| **Dampios** | Performance Evaluation / Quality Analysis | Reconstruction/generation quality, SSIM, PSNR, MSE, and spectral error |

## Repository structure

```text
group_d_vae_common_baseline/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Group_D_Common_VAE_Baseline.ipynb
├── src/
│   └── Group_D_Common_VAE_Baseline.py
├── data/          # MNIST is downloaded here; dataset files are not committed
├── models/        # Common VAE checkpoint is saved here
├── outputs/       # CSV, NPZ, and generated figures/metrics
└── exports/       # HTML exports of individual notebooks
```

## Setup

Create and activate a virtual environment, then install the dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/Group_D_Common_VAE_Baseline.ipynb` and run the cells from top to bottom. The first run downloads MNIST and trains the baseline model. Later runs load the saved checkpoint unless `FORCE_RETRAIN` is changed to `True`.

## Recommended Git workflow

Keep `main` as the frozen common baseline. Each member should work in a separate branch:

```bash
git checkout -b manasan-latent-structure
# or
git checkout -b almazan-frequency-analysis
# or
git checkout -b dampios-performance-evaluation
```

Each member should then copy the shared notebook and rename the copy, for example:

```text
notebooks/Manasan_Latent_Structure.ipynb
notebooks/Almazan_Frequency_Analysis.ipynb
notebooks/Dampios_Performance_Evaluation.ipynb
```

This protects the shared baseline while allowing individual code, visualizations, findings, and conclusions.

## Sharing the exact same model

The notebook saves a small checkpoint in `models/`. After one member completes the common training run, the group may commit that checkpoint so everyone evaluates the same VAE parameters. If the architecture or `LATENT_DIM` changes, use a new checkpoint filename or retrain the model.

## Exporting the individual Python source and annotated HTML

After completing an individual notebook, export the Python source with:

```bash
jupyter nbconvert --to python notebooks/Your_Individual_Notebook.ipynb --output-dir src
```

Then export the annotated HTML with:

```bash
jupyter nbconvert --to html notebooks/Your_Individual_Notebook.ipynb --output-dir exports
```

Both exports should be generated **after** the member has added individual annotations, visualizations, results, and conclusions. The included `src/Group_D_Common_VAE_Baseline.py` is only the source export of the shared baseline.

## Academic separation rule

The dataset, preprocessing, baseline architecture, checkpoint, and reusable helper functions may be shared. The members should not submit identical:

- analyses;
- visualizations;
- research questions;
- interpretations;
- reflections; or
- conclusions.

The final notebooks should clearly state which cells are shared baseline code and which cells are the member's individual contribution.
