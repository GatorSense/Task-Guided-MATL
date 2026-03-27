# Task-Guided Multi-Annotation Triplet Learning (TG-MATL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Task-Guided Multi-Annotation Triplet Learning for Remote Sensing Representations**

This repository implements a task-guided approach to multi-annotation triplet learning (TG-MATL) for learning robust embeddings from remote sensing imagery. The framework compares different loss functions and task-head architectures across multiple embedding types (CLIP, DINOv2, MAE).

## Repository Status

**Cleaned for Reproducibility**
- Removed 936 lines of unused utility functions from `awir_utilities.py` (1146 → 211 lines)
- Removed 152 lines of unused loss functions from `awir_custom_losses.py` (420 → 268 lines)  
- Fixed imports across all experiment scripts to use only available functions
- Standardized to relative paths for cross-platform reproducibility
- Added comprehensive packaging (requirements.txt, setup.py, .gitignore)

## Quick Start

### Prerequisites
- Python 3.8+
- 8-16 GB RAM (for training)
- GPU recommended (NVIDIA with CUDA support)

### Installation

**Option 1: Conda**
```bash
conda env create -f environment.yml
conda activate matl
```

**Option 2: pip**
```bash
pip install -r requirements.txt
```

**Option 3: Development install**
```bash
pip install -e .
```

## Data Requirements

Before running experiments, prepare the following:

### 1. Input Data (`data.npz`)
Create a `data.npz` file in the root directory containing:
```python
import numpy as np

data = {
    'rgb': rgb_images,              # Shape: (N, 300, 300, 3) - RGB images
    'thermal': thermal_images,      # Shape: (N, 300, 300, 1) - Thermal images
    'class_label': class_labels,    # Shape: (N,) - Class labels (0-2)
    'box_label': bounding_boxes,    # Shape: (N, 4) as [xmin, xmax, ymin, ymax]
}
np.savez('data.npz', **data)
```

### 2. Embedding Files
Create `./embeddings/` directory with pre-computed embeddings:
```
embeddings/
├── awir_clip_emb.npy         # CLIP embeddings (N, 512)
├── awir_dinov2_emb.npy       # DINOv2 embeddings (N, 768)
└── awir_mae_emb.npy          # MAE embeddings (N, 768)
```

### 3. External Dependencies
The following modules are imported but not included. They should be provided separately:
- `triplet_loss` — Standard triplet loss implementations
- `dual_triplet_loss_clf_aspect` — Multi-annotation triplet loss  
- `continuous_triplet_loss` — Continuous feature triplet loss
- `TripletNetwork_Online` — Neural network architectures (projection/task heads)

Place these in the root directory alongside the scripts.

## Directory Structure

```
Task-Guided-MATL/
├── matl/                           # Core package (Python module)
│   ├── __init__.py                 # Package initialization & exports
│   ├── awir_utilities.py           # Data processing (211 lines, 4 functions)
│   └── awir_custom_losses.py       # Loss functions (268 lines, 10 functions)
│
├── experiments/                    # Experiment scripts
│   ├── igarss_exp1.py              # Projection head training
│   ├── igarss_exp1_sweep.py        # TG-MATL parameter sweep
│   ├── igarss_exp2.py              # Task head training
│   └── igarss_exp2_sweep.py        # Task heads with sweep parameters
│
├── Configuration & Packaging
│   ├── environment.yml             # Conda environment specification
│   ├── requirements.txt            # pip dependencies
│   ├── setup.py                    # Package setup
│   ├── .gitignore                  # Git patterns (excludes data, results)
│   └── README.md                   # This file
│
├── Input Data (user-provided)
│   ├── data.npz                    # Input data file
│   └── embeddings/                 # Pre-computed embeddings
│       ├── awir_clip_emb.npy
│       ├── awir_dinov2_emb.npy
│       └── awir_mae_emb.npy
│
├── Results (generated)
│   └── results/
│       ├── trained_models/         # Trained projection heads & task heads
│       ├── exp1_test_projections/  # Test embeddings from Exp1
│       ├── exp2_results/           # Task head evaluation metrics
│       └── exp1_timings/           # Training time statistics
```
│   ├── data.npz                    # Input data file
│   └── embeddings/                 # Pre-computed embeddings
│       ├── awir_clip_emb.npy
│       ├── awir_dinov2_emb.npy
│       └── awir_mae_emb.npy
│
└── Results (generated)
    └── results/
        ├── trained_models/         # Trained projection heads & task heads
        ├── exp1_test_projections/  # Test embeddings from Exp1
        ├── exp2_results/           # Task head evaluation metrics
        └── exp1_timings/           # Training time statistics
```

## Core Scripts

### Experiment 1: Projection Head Training (`experiments/igarss_exp1.py`)

Trains projection heads using four loss functions on three embedding types with 8-fold stratified cross-validation:

**Loss Functions:**
- **DTL** (Deep Triplet Loss): Standard triplet loss on class labels
- **DTL-Hard**: Hard negative mining variant
- **MATL** (Multi-Annotation Triplet Loss): Combines class + box triplet losses
- **TG-MATL** (Task-Guided MATL): MI-weighted sample selection

**Embeddings:** CLIP, DINOv2, MAE (1024-dim pre-computed)

```bash
cd experiments
python igarss_exp1.py --margin 0.1 --batch_size 32 --test_size 0.7
cd ..
```

**Outputs:**
- Models: `results/trained_models/exp1_emb_proj/{embedding}/{method}_best_fold{i}.h5`
- Test embeddings: `results/exp1_test_projections/{embedding}/{method}_proj_val0.7_fold{i}.npy`
- Timings: `results/exp1_timings/timings_test0.7.csv`

### Experiment 1 Parameter Sweep (`experiments/igarss_exp1_sweep.py`)

Grid search over TG-MATL hyperparameters:
- `top_percent`: [40%, 50%, 60%, 70%] — high mutual information samples
- `random_percent`: [5%, 10%, 15%, 20%, 25%, 30%] — random samples

```bash
cd experiments
python igarss_exp1_sweep.py --margin 0.1 --batch_size 32 --test_size 0.7
cd ..
```

### Experiment 2: Task Head Training (`experiments/igarss_exp2.py`)

Trains task-specific heads on learned embeddings:
- **Classification head**: 3-way object classifier
- **Regression head (box features)**: Normalized scale/aspect ratio
- **Regression head (box location)**: Normalized center coordinates

Evaluates on:
- Base embeddings (no projection)
- DTL, DTL-Hard, MATL, TG-MATL projections

```bash
cd experiments
python igarss_exp2.py --margin 0.1 --batch_size 32 --test_size 0.7
cd ..
```

**Outputs:**
- Task heads: `results/trained_models/exp2_task_heads/{embedding}/{method}/*_head_run*.h5`
- Metrics: `results/exp2_results/{embedding}/{method}_fold{i}_taskhead_metrics.npy`

### Experiment 2 Parameter Sweep (`experiments/igarss_exp2_sweep.py`)

Task heads using TG-MATL models from `igarss_exp1_sweep.py`:

```bash
cd experiments
python igarss_exp2_sweep.py --margin 0.1 --batch_size 32 --test_size 0.7
cd ..
```

## Utility Modules

### `awir_utilities.py` (211 lines, 4 functions)
**Available functions (only these needed):**

- **`compute_per_sample_metric(y_cls_encoded, y_box_features, metric='mi')`**
  - Computes mutual information or ANOVA between class labels and box features
  - Used for task-guided sample weighting in TG-MATL
  - Returns: per-sample metric scores (N,)

- **`generate_triplet_box_label_normalized(y_box, y_cls, n_clusters=3, ...)`**
  - Creates box triplet labels using KMeans clustering on normalized box geometry
  - Generates 13 normalized geometric features (width, height, area, etc.)
  - Returns: triplet labels (N,), feature matrix (N, 13)

- **`global_max_normalize(data)`**
  - Normalizes image data by dividing by global maximum value
  - Input: (N, H, W, C) array
  - Returns: normalized array in [0, 1]

- **`assign_tile_labels(y_box_cenwh)`**
  - Assigns 3×3 spatial grid labels based on box center coordinates
  - Input: (N, 4) as [x_center, y_center, width, height] (normalized)
  - Returns: tile labels (N,) in range [0, 8]

**Removed functions** (936 lines):
- Comet ML logging (plot_pca_umap, log_3d_pca, log_weight_distribution, etc.)
- Model loading utilities (load_embedding_model, load_mm_embedding_model)
- IoU/mask calculations (calculate_iou, calculate_mask_iou, adjust_and_calculate_iou_mask)
- Resampling (resampling, generate_prototypes_with_cosine_distance)
- Feature weighting (weigh_features, weigh_features_multi, compute_label_weights)
- Unused label generators (generate_triplet_box_label, generate_triplet_box_label_one_hot)

### `awir_custom_losses.py` (268 lines, 10 functions)
**Available loss functions:**

- **`keras_batch_all_triplet_loss(margin=0.0)`** — DTL loss function wrapper
- **`keras_batch_all_triplet_loss_hard(margin=0.0)`** — Hard DTL variant
- **`keras_batch_all_triplet_double_loss(margin=0.0, box_weight=0.5)`** — MATL loss
- **`keras_batch_all_triplet_double_loss_top_random(...)`** — TG-MATL loss with sample selection
- **`compute_selection_mask_tf(labels_clf, sample_metric, top_percent, random_percent)`** — TG-MATL sample selector
- **`keras_batch_all_triplet_continuous_loss_final(margin=0.01)`** — Box feature triplet loss
- **`keras_batch_all_continuous_triplet_loss_final_adjustable_comparisons(num_comparisons)`** — Adjustable comparisons
- **`ciou_loss(y_true, y_pred, alpha=0.5)`** — Complete IoU loss for boxes
- **`dice_loss(y_true, y_pred, smooth=1e-6)`** — Dice loss (segmentation-style)
- **`ssim_loss(alpha=0.1)`** — SSIM + MSE combined loss

**Removed functions** (152 lines):
- keras_batch_all_triplet_continuous_loss_final_weighted variants
- keras_batch_all_continuous_prototype_loss variants
- keras_batch_all_triplet_continuous_loss
- keras_batch_all_triplet_continuous_loss_multimodal
- vae_kl_loss
- multilabel_focal_loss

## Experimental Design

### Cross-Validation
- **8-fold Stratified K-Fold** on all samples
- **Train/Val split**: 30/70 ratio within training folds
- **Test split**: Fixed 30% held out from fold construction
- **Reproducibility**: Random seeds set for deterministic splits

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1-score
- **Regression**: MSE, R² score for box features and location
- **Timing**: Training duration per fold/method

### Output Structure
```
results/
└── exp1_timings/
    └── timings_test0.7.csv  # Per-fold training times
└── trained_models/
    └── exp1_emb_proj/
        ├── clip/{method}_best_fold{i}.h5
        ├── dinov2/{method}_best_fold{i}.h5
        └── mae/{method}_best_fold{i}.h5
    └── exp2_task_heads/
        └── {embedding}/{method}/
            ├── class_head_run{i}.h5
            ├── reg_boxfeat_head_run{i}.h5
            └── reg_boxloc_head_run{i}.h5
└── exp1_test_projections/
    └── {embedding}/{method}_proj_val0.7_fold{i}.npy
└── exp2_results/
    └── {embedding}/{method}_fold{i}_taskhead_metrics.npy
```

## Reproducibility

This repository is designed for maximum reproducibility:

- **Relative Paths** — All paths use `./data/`, `./embeddings/`, `./results/`
- **Fixed Random Seeds** — Set before data splits and model initialization
- **Deterministic CV** — StratifiedKFold with fixed random_state
- **Minimal Dependencies** — requirements.txt pins all package versions
- **Clean Code** — Removed all utility functions not used by experiments
- **Documented Imports** — All dependencies clearly specified  

**To reproduce results:**
1. Install dependencies: `pip install -r requirements.txt`
2. Place `data.npz` and embeddings in appropriate directories
3. Ensure external modules (triplet_loss, TripletNetwork_Online) are available in root directory
4. Run experiments:
   ```bash
   cd experiments
   python igarss_exp1.py --margin 0.1 --batch_size 32 --test_size 0.7
   python igarss_exp2.py --margin 0.1 --batch_size 32 --test_size 0.7
   cd ..
   ```
5. Results saved to `./results/` with proper structure

## Citation

If you use this code, please cite:

```bibtex
@article{zhou2024task,
  title={Task-Guided Multi-Annotation Triplet Learning for Remote Sensing Representations},
  author={Zhou, M. and ...},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact: m.zhou@example.com

---

**Last Updated:** March 2026
**Status:** Cleaned & Packaged for Reproducibility
