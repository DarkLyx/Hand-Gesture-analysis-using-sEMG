import os
import numpy as np

from tensorflow.keras.utils import to_categorical #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score

import common.config as cfg
import common.metrics as met
from dl_approaches.CNN.model_cnn import create_cnn_model
from dl_approaches.CNN.model_vgg_adapter import create_vgg_adapter_model
from dl_approaches.prepare_tensors import generate_tensors

def run_cnn_experiment(mode):
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

    n_folds = cfg.N_FOLDS if hasattr(cfg, 'N_FOLDS') else 5
    gkf = GroupKFold(n_splits=n_folds)
    
    oof_y_true = []
    oof_y_pred = []

    for train_idx, val_idx in gkf.split(X_tv, y_tv, groups=sub_tv):
        X_train, y_train = X_tv[train_idx], y_tv[train_idx]
        X_val, y_val = X_tv[val_idx], y_tv[val_idx]

        if mode == "CNN-VGG-transfert":
            model = create_vgg_adapter_model(X.shape[1:], len(classes))
        else:
            model = create_cnn_model((cfg.WINDOW_SIZE, X.shape[2]), len(classes))

        callbacks = [
            EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=4, verbose=1)
        ]
        
        model.fit(X_train, y_train, 
                  validation_data=(X_val, y_val), 
                  epochs=cfg.EPOCHS, 
                  batch_size=cfg.BATCH_SIZE, 
                  callbacks=callbacks, 
                  verbose=1)

        y_val_pred_fold = np.argmax(model.predict(X_val), axis=1)
        y_val_true_fold = np.argmax(y_val, axis=1)
        
        oof_y_true.extend(y_val_true_fold)
        oof_y_pred.extend(y_val_pred_fold)

    if mode == "CNN-VGG-transfert":
        model = create_vgg_adapter_model(X.shape[1:], len(classes))
    else:
        model = create_cnn_model((cfg.WINDOW_SIZE, X.shape[2]), len(classes))

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=4, verbose=1)
    ]
    
    history = model.fit(X_tv, y_tv, 
                        validation_data=(X_test, y_test), 
                        epochs=cfg.EPOCHS, 
                        batch_size=cfg.BATCH_SIZE, 
                        callbacks=callbacks, 
                        verbose=1)

    met.plot_training_history(history, mode_name=mode)
    
    y_test_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_true = np.argmax(y_test, axis=1)
    
    met.plot_confusion_matrix(y_test_true, y_test_pred, mode_name=mode, dataset_name="Test")

    met.save_comparison_results(
        y_val_true=np.array(oof_y_true), 
        y_val_pred=np.array(oof_y_pred),   
        y_test_true=y_test_true, 
        y_test_pred=y_test_pred, 
        mode_name=mode
    )