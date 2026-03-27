import os
import numpy as np
import time
import tensorflow as tf
import tensorflow.keras as keras
import argparse
from TripletNetwork_Online import projection_head
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import sys
sys.path.insert(0, '..')

from matl import (
    generate_triplet_box_label_normalized,
    global_max_normalize,
    assign_tile_labels,
    compute_per_sample_metric,
    keras_batch_all_triplet_double_loss_top_random,
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class CometCallback(keras.callbacks.Callback):
    def __init__(self, experiment, total_epochs):
        super(CometCallback, self).__init__()
        self.experiment = experiment
        self.global_epoch = 0  # Initialize a global epoch counter
        self.total_epochs = total_epochs  # Total number of epochs to log

    def on_epoch_end(self, epoch, logs=None):
        # Log training and validation loss using the global epoch counter
        self.experiment.log_metric("train_loss", logs["loss"], step=self.global_epoch)
        # self.experiment.log_metric("val_loss", logs["val_loss"], step=self.global_epoch)
        
        # Increment the global epoch counter
        self.global_epoch += 1

                        
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--embedding_dimension", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--modality", type=str, default='rgb')
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--box_weight", type=float, default=0.5)
    args = parser.parse_args()
    
    parameters = {
    "epochs": 5000, 
    "iterations": 1,
    "runs": 1,
    "box_weight": 0.5,
    "embedding_dimension": args.embedding_dimension,
    "margin" : args.margin,
    "batch_size": args.batch_size,
    "test_size": args.test_size,
    "modality": args.modality
}

    # Load the merged data
    f = np.load("data.npz")
    modality_chosen = args.modality
    print('Modality: ' + modality_chosen, flush=True)

    # Extract the necessary arrays
    rgb = f['rgb']  
    thermal = f['thermal']
    y_cls = f['class_label'] 
    y_box = f['box_label'] 
    y_box = np.clip(y_box, None, 300)
    
    y_mask = np.zeros((240, 300, 300, 1))

    # Iterate over each box in y_box and set the corresponding area in y_mask to 1
    for i, (xmin, xmax, ymin, ymax) in enumerate(y_box):
        y_mask[i, ymin:ymax, xmin:xmax, 0] = 1

    #Converting y_box to center, width, height
    
    # Image size (for normalization)
    image_width = 300
    image_height = 300

    # Initialize an array to hold the converted values
    # Shape: (num_samples, 4)
    num_samples = y_box.shape[0]
    y_box_cenwh = np.zeros((num_samples, 4))

    # Calculate center x, center y, width, height and normalize
    for i in range(num_samples):
        xmin, xmax, ymin, ymax = y_box[i]
        xcenter = (xmin + xmax) / 2
        ycenter = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin

        # Normalize by the image size
        y_box_cenwh[i, 0] = xcenter / image_width      # xcenter normalized
        y_box_cenwh[i, 1] = ycenter / image_height     # ycenter normalized
        y_box_cenwh[i, 2] = width / image_width         # width normalized
        y_box_cenwh[i, 3] = height / image_height       # height normalized
        
    # One-hot encode the class labels
    encoder = LabelEncoder()
    one_hot_encoded_classes = encoder.fit_transform(y_cls)

    # y_triplet_box_label, _ = generate_triplet_box_label_one_hot(y_box, y_cls, n_clusters=3)
    
    y_triplet_box_label, features = generate_triplet_box_label_normalized(y_box, y_cls, n_clusters=3)
    
    
    
    # Dictionary to map integers to box descriptions
    box_label_mapping = {
        0: "small elongated box",
        1: "large elongated box",
        2: "small square box"
    }
    
    # Ensure one-hot encoded classes are converted to integer labels
    if len(one_hot_encoded_classes.shape) > 1:
        one_hot_encoded_classes = np.argmax(one_hot_encoded_classes, axis=1)

    # Create a composite label by combining class and triplet labels
    composite_label = [f"{cls}_{triplet}" for cls, triplet in zip(one_hot_encoded_classes, y_triplet_box_label)]

    # Optionally encode the composite labels into integers
    composite_label_encoded = LabelEncoder().fit_transform(composite_label)
        
    rgb_normalized = global_max_normalize(rgb)
    thermal_normalized = global_max_normalize(thermal)
    
    print('rgb_norm min max: ', rgb_normalized.min(), rgb_normalized.max())
    print('thermal_norm min max: ', thermal_normalized.min(), thermal_normalized.max())
  
    test_size = args.test_size
    print('testing size: ' + str(test_size))
    
    tile_labels = assign_tile_labels(y_box_cenwh)
    
    awir_dinov2_emb_loaded = np.load("/blue/azare/zhou.m/dissertation_comparisons/awir_dinov2_emb.npy")
    awir_clip_emb_loaded = np.load("/blue/azare/zhou.m/dissertation_comparisons/awir_clip_emb.npy")
    awir_mae_emb_loaded = np.load("/blue/azare/zhou.m/dissertation_comparisons/awir_mae_emb.npy")

    test_frac = 0.7
    train_frac = 0.3
    
    # First: split off the test set once (fixed)
    (
        X_rgb_trainval, X_rgb_test,
        X_thermal_trainval, X_thermal_test,
        y_cls_trainval, y_cls_test,
        y_box_trainval, y_box_test,
        y_mask_trainval, y_mask_test,
        y_triplet_box_trainval, y_triplet_box_test,
        box_feat_trainval, box_feat_test,
        awir_clip_emb_trainval, awir_clip_emb_test,
        awir_dinov2_emb_trainval, awir_dinov2_emb_test,
        awir_mae_emb_trainval, awir_mae_emb_test,
        composite_label_encoded_trainval, composite_label_encoded_test
    ) = train_test_split(
        rgb_normalized, thermal_normalized, y_cls, y_box_cenwh, y_mask, y_triplet_box_label,
        features, awir_clip_emb_loaded, awir_dinov2_emb_loaded, awir_mae_emb_loaded,
        composite_label_encoded,
        test_size=test_frac, stratify=composite_label_encoded, random_state=42
    )
   


 

    # Cross-validation on the subsample
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

    

    results_summary = []  # Will hold per-run stats for CSV

    # ------------------- CROSS-VALIDATION LOOP -------------------
    # for fold, (train_idx, val_idx) in enumerate(skf.split(X_rgb_sub, composite_label_encoded_sub)):
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_rgb_trainval, y_cls_trainval)):
        print(f"  Fold {fold+1}")

        # --- Data splits ---
        X_rgb_train, X_rgb_val = X_rgb_trainval[train_idx], X_rgb_trainval[val_idx]
        X_thermal_train, X_thermal_val = X_thermal_trainval[train_idx], X_thermal_trainval[val_idx]
        y_cls_train, y_cls_val = y_cls_trainval[train_idx], y_cls_trainval[val_idx]
        y_box_train, y_box_val = y_box_trainval[train_idx], y_box_trainval[val_idx]
        y_mask_train, y_mask_val = y_mask_trainval[train_idx], y_mask_trainval[val_idx]
        y_triplet_box_train, y_triplet_box_val = y_triplet_box_trainval[train_idx], y_triplet_box_trainval[val_idx]
        box_feat_train, box_feat_val = box_feat_trainval[train_idx], box_feat_trainval[val_idx]
        awir_clip_emb_train, awir_clip_emb_val = awir_clip_emb_trainval[train_idx], awir_clip_emb_trainval[val_idx]
        awir_dinov2_emb_train, awir_dinov2_emb_val = awir_dinov2_emb_trainval[train_idx], awir_dinov2_emb_trainval[val_idx]
        awir_mae_emb_train, awir_mae_emb_val = awir_mae_emb_trainval[train_idx], awir_mae_emb_trainval[val_idx]

        print('Number of training samples: ', X_rgb_train.shape[0])
        print('Number of validation samples: ', X_rgb_val.shape[0])

        # --- Encode labels ---
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_cls_train)
        y_val_encoded = label_encoder.transform(y_cls_val)
        y_train_class_categorical = tf.keras.utils.to_categorical(y_train_encoded)
        y_val_class_categorical = tf.keras.utils.to_categorical(y_val_encoded)
        
        # ===== Normalize box_feat using Min–Max from trainval =====
        box_min = box_feat_train.min(axis=0)
        box_max = box_feat_train.max(axis=0)
        box_range = np.where((box_max - box_min) == 0, 1e-8, box_max - box_min)  # prevent divide by zero

        box_feat_train_norm = (box_feat_train - box_min) / box_range
        box_feat_val_norm = (box_feat_val - box_min) / box_range
        box_feat_test_norm = (box_feat_test - box_min) / box_range

        box_feat_train_norm = np.clip(box_feat_train_norm, 0, 1)
        box_feat_val_norm = np.clip(box_feat_val_norm, 0, 1)
        box_feat_test_norm = np.clip(box_feat_test_norm, 0, 1)
        
        # ===== Print min and max for box features =====
        print("box_feat_train_norm range:", box_feat_train_norm.min(), box_feat_train_norm.max())
        print("box_feat_test_norm range:", box_feat_test_norm.min(), box_feat_test_norm.max())
        

        # ------------------ Main Loop ------------------
        for experiment_used in ['clip', 'dinov2', 'mae']:
            print(f"Training {experiment_used} projection", flush=True)

            # Select embeddings
            if experiment_used == 'clip':
                train_embeddings, val_embeddings, test_embeddings = awir_clip_emb_train, awir_clip_emb_val, awir_clip_emb_test
            elif experiment_used == 'dinov2':
                train_embeddings, val_embeddings, test_embeddings = awir_dinov2_emb_train, awir_dinov2_emb_val, awir_dinov2_emb_test
            elif experiment_used == 'mae':
                train_embeddings, val_embeddings, test_embeddings = awir_mae_emb_train, awir_mae_emb_val, awir_mae_emb_test

            input_dim = train_embeddings.shape[1]
            run_number = fold + 1

            # --- Prepare TG_MATL combined features --- 
            sample_metric_train = compute_per_sample_metric(y_train_encoded, y_box_train, metric='mi')
            sample_metric_val   = compute_per_sample_metric(y_val_encoded, y_box_val, metric='mi')

            combined_train_tg_matl = np.vstack((y_train_encoded, y_triplet_box_train, sample_metric_train)).T
            combined_val_tg_matl   = np.vstack((y_val_encoded,   y_triplet_box_val,   sample_metric_val)).T    

            # --- Define GRID SEARCH values ---
            top_percent_grid    = [0.40, 0.50, 0.60, 0.70]
            random_percent_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

            # ------------------- GRID SEARCH -------------------
            for top_p in top_percent_grid:
                for rand_p in random_percent_grid:
                    print(f" ▶ TG_MATL top={top_p} random={rand_p}", flush=True)

                    # Save folder for this configuration
                    base_save_path = (
                        f"/blue/azare/zhou.m/awir/igarss_26_results/"
                        f"30_training/trained_models/exp1_emb_proj/{experiment_used}/tp{int(top_p*100)}_rp{int(rand_p*100)}"
                    )
                    os.makedirs(base_save_path, exist_ok=True)

                    tg_matl_ckpt_path = f"{base_save_path}/tg_matl_best_fold{run_number}.h5"
                    tg_matl_ckpt = ModelCheckpoint(tg_matl_ckpt_path, monitor='val_loss', save_best_only=True, verbose=0)

                    # ------------------- TRAIN TG_MATL -------------------
                    tg_matl_model = projection_head(input_dim)
                    tg_matl_model.compile(
                        optimizer=Adam(0.0001),
                        loss=[keras_batch_all_triplet_double_loss_top_random(
                            margin=parameters['margin'],
                            box_weight=parameters['box_weight'],
                            top_percent=top_p,
                            random_percent=rand_p
                        )]
                    )

                    start_time = time.time()
                    tg_matl_history = tg_matl_model.fit(
                        train_embeddings,
                        combined_train_tg_matl,
                        validation_data=(val_embeddings, combined_val_tg_matl),
                        epochs=parameters['epochs'],
                        batch_size=parameters['batch_size'],
                        verbose=0,
                        callbacks=[tg_matl_ckpt]
                    )
                    duration = time.time() - start_time

                    # ------------------- PROJECTIONS -------------------
                    proj_model = tf.keras.Model(
                        inputs=tg_matl_model.input,
                        outputs=tg_matl_model.get_layer("proj").output
                    )

                    tg_matl_test_proj = proj_model.predict(test_embeddings, verbose=0)

                    emb_dir = (
                        f"/blue/azare/zhou.m/awir/igarss_26_results/"
                        f"30_training/exp1_test_projections/{experiment_used}/tp{int(top_p*100)}_rp{int(rand_p*100)}"
                    )
                    os.makedirs(emb_dir, exist_ok=True)

                    np.save(
                        f"{emb_dir}/tg_matl_proj_val{test_frac}_fold{run_number}.npy",
                        tg_matl_test_proj
                    )

                    # ------------------- SAVE PER-RUN RESULTS -------------------
                    results_summary.append({
                        "experiment": experiment_used,
                        "fold": run_number,
                        "top_percent": top_p,
                        "random_percent": rand_p,
                        "train_time": duration
                    })

        # ------------------- SAVE SUMMARY CSV -------------------
        summary_df = pd.DataFrame(results_summary)
        summary_dir = "/blue/azare/zhou.m/awir/igarss_26_results/30_training/exp1_timings"
        os.makedirs(summary_dir, exist_ok=True)

        summary_csv_path = os.path.join(summary_dir, f"timings_test{test_frac}_sweep.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\n✅ Saved summary CSV to {summary_csv_path}")