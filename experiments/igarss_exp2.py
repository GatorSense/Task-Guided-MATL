import os
import numpy as np
import time
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from TripletNetwork_Online import projection_head, class_head, regression_head
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import pandas as pd
import sys
sys.path.insert(0, '..')

from matl import (
    generate_triplet_box_label_normalized,
    global_max_normalize,
    assign_tile_labels,
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
    "embedding_dimension": args.embedding_dimension,
    "margin" : args.margin,
    "batch_size": args.batch_size,
    "test_size": args.test_size
}

    # Load the merged data
    f = np.load("data.npz")


    # Extract the necessary arrays
    rgb = f['rgb']  
    thermal = f['thermal']
    y_cls = f['class_label'] 
    y_box = f['box_label'] 
    y_box = np.clip(y_box, None, 300)
    
    label_encoder = LabelEncoder()
    y_cls_encoded = label_encoder.fit_transform(y_cls)

    
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
    
    awir_dinov2_emb_loaded = np.load("./embeddings/awir_dinov2_emb.npy")
    awir_clip_emb_loaded = np.load("./embeddings/awir_clip_emb.npy")
    awir_mae_emb_loaded = np.load("./embeddings/awir_mae_emb.npy")

    
    val_frac = 0.7
    train_frac = 0.3
    

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
        test_size=val_frac, stratify=composite_label_encoded, random_state=42
    )
    



    


    # Cross-validation on the subsample
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    


    # ===== Directories =====
    base_save_root = "./results/"
    model_save_root = os.path.join(base_save_root, "trained_models/exp2_task_heads")

    os.makedirs(model_save_root, exist_ok=True)

    
    
    # To store timing and results
    timing_stats = {m: {"base": [], "dtl": [], "dtl_hard": [], "matl": [], "tg_matl": []} for m in ["clip", "dinov2", "mae"]}
    results_stats = {m: {"base": [], "dtl": [], "dtl_hard": [], "matl": [], "tg_matl": []} for m in ["clip", "dinov2", "mae"]}




    results_summary = []  # Will hold per-run stats for CSV
    

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_rgb_trainval, composite_label_encoded_trainval)):
        print(f"=== Fold {fold+1} ===")
        
        # Train/val split
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


        print('Number of training samples:', X_rgb_train.shape[0])
        print('Number of validation samples:', X_rgb_val.shape[0])
        


        # Encode class labels
        y_train_encoded = label_encoder.fit_transform(y_cls_train)
        y_val_encoded = label_encoder.transform(y_cls_val)
        y_test_encoded = label_encoder.transform(y_cls_test)

        y_train_class_categorical = tf.keras.utils.to_categorical(y_train_encoded)
        y_val_class_categorical = tf.keras.utils.to_categorical(y_val_encoded)

        box_loc_train = y_box_train[:, :2]
        box_loc_val = y_box_val[:, :2]   
        box_loc_test = y_box_test[:, :2]
        
        
        # ===== Print min and max for box features =====
        print("box_feat_train_norm range:", box_feat_train_norm.min(), box_feat_train_norm.max())
        print("box_feat_test_norm range:", box_feat_test_norm.min(), box_feat_test_norm.max())

        print("box_loc_train range:", box_loc_train.min(), box_loc_train.max())
        print("box_loc_test range:", box_loc_test.min(), box_loc_test.max())


        for experiment_used in ["clip", "dinov2", "mae"]:
            print(f"--- {experiment_used.upper()} ---", flush=True)
            run_number = fold + 1
            
            # Select base embeddings
            if experiment_used == 'clip':
                base_train_embeddings, base_val_embeddings, base_test_embeddings = awir_clip_emb_train, awir_clip_emb_val, awir_clip_emb_test
            elif experiment_used == 'dinov2':
                base_train_embeddings, base_val_embeddings, base_test_embeddings = awir_dinov2_emb_train, awir_dinov2_emb_val, awir_dinov2_emb_test
            elif experiment_used == 'mae':
                base_train_embeddings, base_val_embeddings, base_test_embeddings = awir_mae_emb_train, awir_mae_emb_val, awir_mae_emb_test
                
            # --- Train Base Embedding Task Heads ONCE ---
            print("Training task heads on BASE embeddings")
            input_dim = base_train_embeddings.shape[1]
            run_dir = os.path.join(model_save_root, experiment_used, "base")
            os.makedirs(run_dir, exist_ok=True)

            # Class Head
            class_model = class_head(input_dim, 3)
            class_model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
            class_ckpt = os.path.join(run_dir, f"class_head_run_base.h5")
            class_model.fit(
                base_train_embeddings, y_train_class_categorical,
                validation_data=(base_val_embeddings, y_val_class_categorical),
                epochs=parameters['epochs'],
                batch_size=parameters['batch_size'],
                verbose=0,
                callbacks=[ModelCheckpoint(class_ckpt, save_best_only=True, monitor='val_loss', verbose=0)]
            )
            class_model.load_weights(class_ckpt)

            # Regression Head (Box Features)
            reg_boxfeat_model = regression_head(input_dim, box_feat_train_norm.shape[1])
            reg_boxfeat_model.compile(optimizer=Adam(1e-4), loss='mse')
            reg_boxfeat_ckpt = os.path.join(run_dir, f"reg_boxfeat_head_run_base.h5")
            reg_boxfeat_model.fit(
                base_train_embeddings, box_feat_train_norm,
                validation_data=(base_val_embeddings, box_feat_val_norm),
                epochs=parameters['epochs'],
                batch_size=parameters['batch_size'],
                verbose=0,
                callbacks=[ModelCheckpoint(reg_boxfeat_ckpt, save_best_only=True, monitor='val_loss', verbose=0)]
            )
            reg_boxfeat_model.load_weights(reg_boxfeat_ckpt)

            # Regression Head (Box Location)
            reg_boxloc_model = regression_head(input_dim, box_loc_train.shape[1])
            reg_boxloc_model.compile(optimizer=Adam(1e-4), loss='mse')
            reg_boxloc_ckpt = os.path.join(run_dir, f"reg_boxloc_head_run_base.h5")
            reg_boxloc_model.fit(
                base_train_embeddings, box_loc_train,
                validation_data=(base_val_embeddings, box_loc_val),
                epochs=parameters['epochs'],
                batch_size=parameters['batch_size'],
                verbose=0,
                callbacks=[ModelCheckpoint(reg_boxloc_ckpt, save_best_only=True, monitor='val_loss', verbose=0)]
            )
            reg_boxloc_model.load_weights(reg_boxloc_ckpt)

            # Evaluate on base test set
            y_pred_cls = np.argmax(class_model.predict(base_test_embeddings, verbose=0), axis=1)
            acc = accuracy_score(y_test_encoded, y_pred_cls)
            f1 = f1_score(y_test_encoded, y_pred_cls, average='weighted')
            pred_box_feat = reg_boxfeat_model.predict(base_test_embeddings, verbose=0)
            pred_box_loc = reg_boxloc_model.predict(base_test_embeddings, verbose=0)
            mse_feat = mean_squared_error(box_feat_test_norm, pred_box_feat)
            r2_feat = r2_score(box_feat_test_norm, pred_box_feat)
            mse_loc = mean_squared_error(box_loc_test, pred_box_loc)
            r2_loc = r2_score(box_loc_test, pred_box_loc)
            
            
            
            print(f"BASE embeddings | Acc={acc:.3f}, F1={f1:.3f}, MSE_feat={mse_feat:.3f}, R2_feat={r2_feat:.3f}, MSE_loc={mse_loc:.3f}, R2_loc={r2_loc:.3f}")

            # Save base results
            results_save_dir = os.path.join("./results/exp2_results", experiment_used)
            os.makedirs(results_save_dir, exist_ok=True)
            
            np.save(os.path.join(results_save_dir, f"base_fold{run_number}_taskhead_metrics.npy"),
                    {"acc": acc, "f1": f1, "mse_feat": mse_feat, "r2_feat": r2_feat, "mse_loc": mse_loc, "r2_loc": r2_loc})


            # Paths to saved models
            dtl_model_path = f"./results/trained_models/exp1_emb_proj/{experiment_used}/dtl_best_fold{run_number}.h5"
        
            dtl_hard_model_path = f"./results/trained_models/exp1_emb_proj/{experiment_used}/dtl_hard_best_fold{run_number}.h5"
            
            matl_model_path = f"./results/trained_models/exp1_emb_proj/{experiment_used}/matl_best_fold{run_number}.h5"
            tg_matl_model_path = f"./results/trained_models/exp1_emb_proj/{experiment_used}/tg_matl_30_10_best_fold{run_number}.h5"

            # Load DTL model and extract proj layer
            dtl_model = projection_head(base_train_embeddings.shape[1])
            dtl_model.load_weights(dtl_model_path)
            dtl_proj_model = tf.keras.Model(inputs=dtl_model.input, outputs=dtl_model.get_layer("proj").output)
            
            # Load DTL_HARD model and extract proj layer
            dtl_hard_model = projection_head(base_train_embeddings.shape[1])
            dtl_hard_model.load_weights(dtl_hard_model_path)
            dtl_hard_proj_model = tf.keras.Model(inputs=dtl_hard_model.input, outputs=dtl_hard_model.get_layer("proj").output)
            
            # Load MATL model and extract proj layer
            matl_model = projection_head(base_train_embeddings.shape[1])
            matl_model.load_weights(matl_model_path)
            matl_proj_model = tf.keras.Model(inputs=matl_model.input, outputs=matl_model.get_layer("proj").output)
            
            # Load TG_MATL model and extract proj layer
            tg_matl_model = projection_head(base_train_embeddings.shape[1])
            tg_matl_model.load_weights(tg_matl_model_path)
            tg_matl_proj_model = tf.keras.Model(inputs=tg_matl_model.input, outputs=tg_matl_model.get_layer("proj").output)


            # Generate embeddings from base embeddings
            dtl_train_embeddings = dtl_proj_model.predict(base_train_embeddings, verbose=0)
            dtl_val_embeddings   = dtl_proj_model.predict(base_val_embeddings, verbose=0)
            dtl_test_embeddings  = dtl_proj_model.predict(base_test_embeddings, verbose=0)

            dtl_hard_train_embeddings = dtl_hard_proj_model.predict(base_train_embeddings, verbose=0)
            dtl_hard_val_embeddings   = dtl_hard_proj_model.predict(base_val_embeddings, verbose=0)
            dtl_hard_test_embeddings  = dtl_hard_proj_model.predict(base_test_embeddings, verbose=0)
            
            matl_train_embeddings = matl_proj_model.predict(base_train_embeddings, verbose=0)
            matl_val_embeddings   = matl_proj_model.predict(base_val_embeddings, verbose=0)
            matl_test_embeddings  = matl_proj_model.predict(base_test_embeddings, verbose=0)
            
            tg_matl_train_embeddings = tg_matl_proj_model.predict(base_train_embeddings, verbose=0)
            tg_matl_val_embeddings   = tg_matl_proj_model.predict(base_val_embeddings, verbose=0)
            tg_matl_test_embeddings  = tg_matl_proj_model.predict(base_test_embeddings, verbose=0)



            # Dictionary of embeddings for the current fold
            emb_dict = {
                "dtl": dtl_train_embeddings, 
                "dtl_hard": dtl_hard_train_embeddings,
                "matl": matl_train_embeddings,
                "tg_matl": tg_matl_train_embeddings
            }

            # Loop over embedding types
            for emb_type, emb_data_train in emb_dict.items():
                print(f"Training task heads on {emb_type.upper()} embeddings")

                if emb_type == "dtl":
                    emb_data_val = dtl_val_embeddings
                    emb_data_test = dtl_test_embeddings
                elif emb_type == "dtl_hard":
                    emb_data_val = dtl_hard_val_embeddings
                    emb_data_test = dtl_hard_test_embeddings
                elif emb_type == "matl":
                    emb_data_val = matl_val_embeddings
                    emb_data_test = matl_test_embeddings
                elif emb_type == "tg_matl":
                    emb_data_val = tg_matl_val_embeddings
                    emb_data_test = tg_matl_test_embeddings
                    
                input_dim = emb_data_train.shape[1]
                run_dir = os.path.join(model_save_root, experiment_used, emb_type)
                os.makedirs(run_dir, exist_ok=True)

                # ===== Class Head =====
                class_model = class_head(input_dim, 3)
                class_model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
                class_ckpt = os.path.join(run_dir, f"class_head_run{run_number}.h5")

                start_time = time.time()
                class_model.fit(
                    emb_data_train, y_train_class_categorical,
                    validation_data=(emb_data_val, y_val_class_categorical),
                    epochs=parameters['epochs'],
                    batch_size=parameters['batch_size'],
                    verbose=0,
                    callbacks=[ModelCheckpoint(class_ckpt, save_best_only=True, monitor='val_loss', verbose=0)]
                )
                train_time_class = time.time() - start_time
                print(f"Class head trained in {train_time_class:.2f}s")

                # ===== Regression Head (Box Features) =====
                reg_boxfeat_model = regression_head(input_dim, box_feat_train_norm.shape[1])
                reg_boxfeat_model.compile(optimizer=Adam(1e-4), loss='mse')
                reg_boxfeat_ckpt = os.path.join(run_dir, f"reg_boxfeat_head_run{run_number}.h5")

                start_time = time.time()
                reg_boxfeat_model.fit(
                    emb_data_train, box_feat_train_norm,
                    validation_data=(emb_data_val, box_feat_val_norm),
                    epochs=parameters['epochs'],
                    batch_size=parameters['batch_size'],
                    verbose=0,
                    callbacks=[ModelCheckpoint(reg_boxfeat_ckpt, save_best_only=True, monitor='val_loss', verbose=0)]
                )
                train_time_boxfeat = time.time() - start_time
                print(f"Box feature head trained in {train_time_boxfeat:.2f}s")

                # ===== Regression Head (Box Location) =====
                reg_boxloc_model = regression_head(input_dim, box_loc_train.shape[1])
                reg_boxloc_model.compile(optimizer=Adam(1e-4), loss='mse')
                reg_boxloc_ckpt = os.path.join(run_dir, f"reg_boxloc_head_run{run_number}.h5")

                start_time = time.time()
                reg_boxloc_model.fit(
                    emb_data_train, box_loc_train,
                    validation_data=(emb_data_val, box_loc_val),
                    epochs=parameters['epochs'],
                    batch_size=parameters['batch_size'],
                    verbose=0,
                    callbacks=[ModelCheckpoint(reg_boxloc_ckpt, save_best_only=True, monitor='val_loss', verbose=0)]
                )
                train_time_boxloc = time.time() - start_time
                print(f"Box location head trained in {train_time_boxloc:.2f}s")

                # ===== Save training times =====
                timing_stats[experiment_used][emb_type].append({
                    "run" : run_number,
                    "class_time": train_time_class,
                    "reg_boxfeat_time": train_time_boxfeat,
                    "reg_boxloc_time": train_time_boxloc
                })

                # ===== Reload best models =====
                class_model.load_weights(class_ckpt)
                reg_boxfeat_model.load_weights(reg_boxfeat_ckpt)
                reg_boxloc_model.load_weights(reg_boxloc_ckpt)

                # ===== Evaluate on test set =====
                y_pred_cls = np.argmax(class_model.predict(emb_data_test, verbose=0), axis=1)
                y_true_cls = y_test_encoded

                acc = accuracy_score(y_true_cls, y_pred_cls)
                f1 = f1_score(y_true_cls, y_pred_cls, average='weighted')

                pred_box_feat = reg_boxfeat_model.predict(emb_data_test, verbose=0)
                pred_box_loc = reg_boxloc_model.predict(emb_data_test, verbose=0)

                mse_feat = mean_squared_error(box_feat_test_norm, pred_box_feat)
                r2_feat = r2_score(box_feat_test_norm, pred_box_feat)
                mse_loc = mean_squared_error(box_loc_test, pred_box_loc)
                r2_loc = r2_score(box_loc_test, pred_box_loc)

                # ===== Save results =====

                np.save(os.path.join(results_save_dir, f"{emb_type}_fold{run_number}_taskhead_metrics.npy"),
                        {"acc": acc, "f1": f1, "mse_feat": mse_feat, "r2_feat": r2_feat, "mse_loc": mse_loc, "r2_loc": r2_loc, "class_time": train_time_class, "reg_boxfeat_time": train_time_boxfeat, "reg_boxloc_time": train_time_boxloc})



