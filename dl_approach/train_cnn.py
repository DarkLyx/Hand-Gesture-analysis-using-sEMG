import os
import numpy as np
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import GroupShuffleSplit

import common.config as cfg
import common.metrics as met
from dl_approach.model import create_cnn_model
from dl_approach.prepare_tensors import generate_tensors

def run_dl_experiment():
    # Tenseurs
    if not os.path.exists(os.path.join(cfg.TENSOR_DIR, 'X.npy')):
        generate_tensors()
    
    X = np.load(os.path.join(cfg.TENSOR_DIR, 'X.npy'))
    y = np.load(os.path.join(cfg.TENSOR_DIR, 'y.npy'))
    subs = np.load(os.path.join(cfg.TENSOR_DIR, 'sub.npy'))

    # Labels
    classes = np.unique(y)
    class_map = {c: i for i, c in enumerate(classes)}
    y_mapped = np.array([class_map[lbl] for lbl in y])
    y_hot = to_categorical(y_mapped, len(classes))

    # Splits
    split_test = GroupShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    tv_idx, test_idx = next(split_test.split(X, y, groups=subs))
    X_tv, y_tv, sub_tv = X[tv_idx], y_hot[tv_idx], subs[tv_idx]
    X_test, y_test = X[test_idx], y_hot[test_idx]

    split_val = GroupShuffleSplit(n_splits=1, test_size=cfg.VALIDATION_SIZE, random_state=cfg.RANDOM_STATE)
    train_idx, val_idx = next(split_val.split(X_tv, y_tv, groups=sub_tv))
    X_train, y_train = X_tv[train_idx], y_tv[train_idx]
    X_val, y_val = X_tv[val_idx], y_tv[val_idx]

    # Train
    model = create_cnn_model((cfg.WINDOW_SIZE, X.shape[2]), len(classes))
    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=4, verbose=1)
    ]
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                        epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, callbacks=callbacks, verbose=1)

    # Eval
    print("\n   [DL] Calcul des scores finaux...")
    met.plot_training_history(history)
    
    # 1. Validation
    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    y_val_true = np.argmax(y_val, axis=1)
    met.plot_confusion_matrix(y_val_true, y_val_pred, mode_name="DL", dataset_name="Validation")

    # 2. Test
    y_test_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_true = np.argmax(y_test, axis=1)
    met.plot_confusion_matrix(y_test_true, y_test_pred, mode_name="DL", dataset_name="Test")

    # --- COMPARAISON VISUELLE (MODIFICATION ICI) ---
    met.save_comparison_results(
        y_val_true, y_val_pred,   # Données Validation
        y_test_true, y_test_pred, # Données Test
        mode_name="DL"
    )