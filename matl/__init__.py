"""
Task-Guided Multi-Annotation Triplet Learning (TG-MATL)

Utilities for learning robust embeddings from remote sensing imagery.
"""

from matl.awir_utilities import (
    compute_per_sample_metric,
    generate_triplet_box_label_normalized,
    global_max_normalize,
    assign_tile_labels,
)

from matl.awir_custom_losses import (
    compute_selection_mask_tf,
    keras_batch_all_triplet_loss,
    keras_batch_all_triplet_loss_hard,
    keras_batch_all_triplet_double_loss,
    keras_batch_all_triplet_double_loss_top_random,
    keras_batch_all_triplet_continuous_loss_final,
    keras_batch_all_continuous_triplet_loss_final_adjustable_comparisons,
    ciou_loss,
    dice_loss,
    ssim_loss,
)

__version__ = "1.0.0"
__all__ = [
    "compute_per_sample_metric",
    "generate_triplet_box_label_normalized",
    "global_max_normalize",
    "assign_tile_labels",
    "compute_selection_mask_tf",
    "keras_batch_all_triplet_loss",
    "keras_batch_all_triplet_loss_hard",
    "keras_batch_all_triplet_double_loss",
    "keras_batch_all_triplet_double_loss_top_random",
    "keras_batch_all_triplet_continuous_loss_final",
    "keras_batch_all_continuous_triplet_loss_final_adjustable_comparisons",
    "ciou_loss",
    "dice_loss",
    "ssim_loss",
]
