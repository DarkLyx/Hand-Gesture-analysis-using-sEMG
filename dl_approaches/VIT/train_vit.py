import os
import numpy as np

from tensorflow.keras.utils import to_categorical #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score

import common.config as cfg
import common.metrics as met
from dl_approaches.VIT.model_vit import create_vit_model
from dl_approaches.prepare_tensors import generate_tensors

import common.postprocessing as post

def run_vit_experiment():
    generate_tensors()

    X = np.load(os.path.join(cfg.TENSOR_DIR, "X.npy"))
    y = np.load(os.path.join(cfg.TENSOR_DIR, "y.npy"))
    subs = np.load(os.path.join(cfg.TENSOR_DIR, "sub.npy"))

    classes = np.unique(y)
    class_map = {c: i for i, c in enumerate(classes)}

    y_mapped = np.array([class_map[lbl] for lbl in y])
    y_hot = to_categorical(y_mapped, len(classes))

    split_test = GroupShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    tv_idx, test_idx = next(split_test.split(X, y, groups=subs))

    X_tv, y_tv, sub_tv = X[tv_idx], y_hot[tv_idx], subs[tv_idx]
    X_test, y_test = X[test_idx], y_hot[test_idx]

    gkf = GroupKFold(n_splits=cfg.N_FOLDS if hasattr(cfg, 'N_FOLDS') else 5)
    val_scores = []
    
    oof_y_true = []
    oof_y_pred = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_tv, y_tv, groups=sub_tv), start=1):
        print(f"\n--- FOLD {fold} ---")
        X_train, y_train = X_tv[train_idx], y_tv[train_idx]
        X_val, y_val = X_tv[val_idx], y_tv[val_idx]

        model = create_vit_model(
            input_shape=(cfg.WINDOW_SIZE, X.shape[2]),
            num_classes=len(classes)
        )

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg.EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        y_val_pred_fold = np.argmax(model.predict(X_val), axis=1)
        y_val_true_fold = np.argmax(y_val, axis=1)

        oof_y_true.extend(y_val_true_fold)
        oof_y_pred.extend(y_val_pred_fold)

        acc = accuracy_score(y_val_true_fold, y_val_pred_fold)
        val_scores.append(acc)
        print(f"-> Accuracy Fold {fold}: {acc:.2%}")

    print(f"\n[VIT] Accuracy CV: {np.mean(val_scores):.2%} (+/- {np.std(val_scores):.2%})")

    model = create_vit_model(
        input_shape=(cfg.WINDOW_SIZE, X.shape[2]),
        num_classes=len(classes)
    )

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)
    ]

    history = model.fit(
        X_tv, y_tv,
        validation_data=(X_test, y_test),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    y_test_pred_raw = np.argmax(model.predict(X_test), axis=1)
    y_test_true = np.argmax(y_test, axis=1)

    y_val_true_final = np.array(oof_y_true)
    y_val_pred_raw_final = np.array(oof_y_pred)
    

    if cfg.USE_POST_PROCESSING:
        print(f"\n[VIT] Applying Majority Voting optimization")
        y_val_final, y_test_final, _ = post.optimize_and_apply_majority_voting(
            y_val_true_final, 
            y_val_pred_raw_final, 
            y_test_pred_raw
        )
        suffix = "_w_post"
    else:
        print(f"\n[VIT] Post-processing DISABLED via config.")
        y_val_final = y_val_pred_raw_final
        y_test_final = y_test_pred_raw
        suffix = "_no_post"

    run_name = "VIT" + suffix
    met.plot_training_history(history, mode_name=run_name)

    met.plot_confusion_matrix(y_val_true_final, y_val_final, mode_name=run_name, dataset_name="Validation")
    met.plot_confusion_matrix(y_test_true, y_test_final, mode_name=run_name, dataset_name="Test")

    met.save_comparison_results(
        y_val_true=y_val_true_final,
        y_val_pred=y_val_final,
        y_test_true=y_test_true, 
        y_test_pred=y_test_final,
        mode_name=run_name
    )