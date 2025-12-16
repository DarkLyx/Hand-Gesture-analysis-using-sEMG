import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import GroupShuffleSplit

import common.config as cfg
import common.metrics as met
from dl_approaches.model_vit import create_vit_model
from dl_approaches.prepare_tensors import generate_tensors

def run_vit_experiment():
    
    if not os.path.exists(os.path.join(cfg.TENSOR_DIR, 'X.npy')):
        generate_tensors()
    
    X = np.load(os.path.join(cfg.TENSOR_DIR, 'X.npy'))
    y = np.load(os.path.join(cfg.TENSOR_DIR, 'y.npy'))
    subs = np.load(os.path.join(cfg.TENSOR_DIR, 'sub.npy'))

    classes = np.unique(y)
    class_map = {c: i for i, c in enumerate(classes)}
    y_mapped = np.array([class_map[lbl] for lbl in y])
    y_hot = to_categorical(y_mapped, len(classes))

    split_test = GroupShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    tv_idx, test_idx = next(split_test.split(X, y, groups=subs))
    X_tv, y_tv, sub_tv = X[tv_idx], y_hot[tv_idx], subs[tv_idx]
    X_test, y_test = X[test_idx], y_hot[test_idx]

    split_val = GroupShuffleSplit(n_splits=1, test_size=cfg.VALIDATION_SIZE, random_state=cfg.RANDOM_STATE)
    train_idx, val_idx = next(split_val.split(X_tv, y_tv, groups=sub_tv))
    X_train, y_train = X_tv[train_idx], y_tv[train_idx]
    X_val, y_val = X_tv[val_idx], y_tv[val_idx]

    
    model = create_vit_model(input_shape=(cfg.WINDOW_SIZE, X.shape[2]), num_classes=len(classes))
    model.summary()

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=cfg.EPOCHS + 10, 
        batch_size=cfg.BATCH_SIZE, 
        callbacks=callbacks, 
        verbose=1
    )

    print("\n   [VIT] Génération des résultats...")
    met.plot_training_history(history, mode_name="VIT")
    
    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    y_val_true = np.argmax(y_val, axis=1)
    
    y_test_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_true = np.argmax(y_test, axis=1)
    
    met.plot_confusion_matrix(y_val_true, y_val_pred, mode_name="VIT", dataset_name="Validation")
    met.plot_confusion_matrix(y_test_true, y_test_pred, mode_name="VIT", dataset_name="Test")

    met.save_comparison_results(
        y_val_true, y_val_pred,
        y_test_true, y_test_pred,
        mode_name="VIT"
        )