# Is Effective Rank a Reliable Predictor of Transfer Performance in Self-Supervised Medical Foundation Models?

This repository contains the code and experiments for a project examining whether geometric metrics (effective rank, participation ratio, spectral decay slope, NAD anisotropy) can predict transfer performance under domain shift and whether Adaptive Collapse Recovery (ACR) can mitigate representation collapse.

## Models and Datasets

- **Models**: ResNet‑50 (supervised), DINOv2 (self‑distilled ViT), MAE (masked autoencoder)
- **Source domain**: RetinaMNIST (diabetic retinopathy grading, 5 classes)
- **Target domain**: IDRiD (different population, camera system, acquisition protocol)

## Key Findings

- DINOv2 achieves the highest transfer performance (balanced accuracy ≈0.30) across all fine‑tuning strategies, even with a frozen backbone.
- Effective rank alone does not reliably predict transferability: ResNet‑50 has the highest source effective rank but transfers poorly, while DINOv2 transfers well despite lower rank.
- ACR (adaptive or fixed‑λ) does not meaningfully improve performance over standard fine‑tuning under this domain shift.
- MAE and ResNet‑50 perform near the random baseline (0.20) on the target task.

## Repository Structure

- `figures/` – generated plots (effective rank comparison, fine‑tuning results, ACR improvement)
- `metrics/` – CSV files with geometric metrics and fine‑tuning results
- `Representation_Collapse_SSL_ACR_V1.ipynb` – main Jupyter notebook (all code cells)

## How to Reproduce

Run the notebook cells sequentially in Google Colab with Google Drive mounted. The notebook will automatically install dependencies, download datasets (RetinaMNIST via MedMNIST, IDRiD via KaggleHub), extract features, compute geometric metrics, run fine‑tuning experiments, and generate tables and figures.

All results are saved to `metrics/` and figures to `figures/` in your Google Drive project directory.

## Requirements

- Python 3.8+
- PyTorch 2.11.0
- timm, transformers, medmnist, scikit‑learn, seaborn, pandas, matplotlib, tqdm, kagglehub

See `requirements.txt` for exact versions.

## Conclusion

DINOv2’s self‑distilled features are highly transferable to medical imaging. Effective rank is not a reliable predictor of transfer success; feature‑task alignment is more critical. ACR shows no clear benefit under the tested conditions.
