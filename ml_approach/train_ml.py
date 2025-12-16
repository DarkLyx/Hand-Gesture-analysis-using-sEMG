import os
import sys
import glob
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

import common.config as cfg
import common.preprocessing as pp
import common.metrics as met
import ml_approach.features as ft
import ml_approach.model as ml_model

def load_or_extract_features():
    if not os.path.exists(cfg.CSV_DIR):
        os.makedirs(cfg.CSV_DIR)

    if os.path.exists(cfg.CSV_FILE):
        print(f"   [ML] Chargement de {cfg.CSV_FILE}...")
        return pd.read_csv(cfg.CSV_FILE)
    
    print("   [ML] Extraction des features (TD+FD)...")
    all_data = []
    
    for subject_id in range(1, 37):
        path = os.path.join(cfg.ROOT_DIR, f"{subject_id:02d}")
        if not os.path.exists(path): continue
        
        for f in glob.glob(os.path.join(path, "*.txt")):
            try:
                df = pd.read_csv(f, sep='\t')
                raw = df[[c for c in df.columns if 'channel' in c]].values
                df[[c for c in df.columns if 'channel' in c]] = pp.butter_bandpass_filter(pp.notch_filter(raw))
                df = pp.clean_and_trim_data(df)
                if df is None: continue
                
                for seg, label in pp.get_windows(df):
                    feats = ft.calculate_features(seg)
                    feats['label'] = label
                    feats['subject_id'] = subject_id
                    all_data.append(feats)
            except: pass
        print(f"   -> Sujet {subject_id} traité")
    
    if not all_data:
        print(f"ERREUR: Aucune donnée trouvée dans {cfg.ROOT_DIR}")
        sys.exit()

    df = pd.DataFrame(all_data)
    print(f"   [ML] Sauvegarde dans {cfg.CSV_FILE}...")
    df.to_csv(cfg.CSV_FILE, index=False)
    return df

def run_ml_experiment():
    df = load_or_extract_features()
    X = df.drop(columns=['label', 'subject_id'])
    y = df['label']
    subjects = df['subject_id']

    split_test = GroupShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    tv_idx, test_idx = next(split_test.split(X, y, groups=subjects))
    X_tv, y_tv, sub_tv = X.iloc[tv_idx], y.iloc[tv_idx], subjects.iloc[tv_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    split_val = GroupShuffleSplit(n_splits=1, test_size=cfg.VALIDATION_SIZE, random_state=cfg.RANDOM_STATE)
    train_idx, val_idx = next(split_val.split(X_tv, y_tv, groups=sub_tv))
    X_train, y_train = X_tv.iloc[train_idx], y_tv.iloc[train_idx]
    X_val, y_val = X_tv.iloc[val_idx], y_tv.iloc[val_idx]

    best_model = ml_model.train_ml_model(X_train, y_train)

    # Optimisation Validation
    y_val_raw = best_model.predict(X_val)
    best_win, best_acc = 1, 0
    for w in [1, 3, 5, 7, 9]:
        acc = accuracy_score(y_val, met.majority_voting(y_val_raw, w))
        if acc > best_acc: best_win, best_acc = w, acc
    print(f"   -> Meilleure fenêtre : {best_win}")

    print("\n   [ML] Calcul des scores finaux...")
    
    y_val_smooth = met.majority_voting(y_val_raw, window_size=best_win)
    met.plot_confusion_matrix(y_val, y_val_smooth, mode_name="ML", dataset_name="Validation")
    y_pred_raw = best_model.predict(X_test)
    y_pred_smooth = met.majority_voting(y_pred_raw, window_size=best_win)
    met.plot_confusion_matrix(y_test, y_pred_smooth, mode_name="ML", dataset_name="Test")

    met.save_comparison_results(
        y_val, y_val_smooth,    
        y_test, y_pred_smooth, 
        mode_name="ML"
    )