import sys
import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import common.config as cfg
import common.preprocessing as pp

def generate_tensors():
    X_list, y_list, sub_list = [], [], []
    print("   [DL] Génération des Tenseurs (Filtres + Z-Score)...")

    for subject_id in range(1, 37):
        path = os.path.join(cfg.ROOT_DIR, f"{subject_id:02d}")
        if not os.path.exists(path): continue
        
        for f in glob.glob(os.path.join(path, "*.txt")):
            try:
                df = pd.read_csv(f, sep='\t')
                chans = [c for c in df.columns if 'channel' in c]
                
                # A. Filtres (Notch + Bandpass)
                df[chans] = pp.butter_bandpass_filter(pp.notch_filter(df[chans].values))
                
                # B. Nettoyage
                df = pp.clean_and_trim_data(df)
                if df is None: continue
                
                # C. Z-SCORE
                scaler = StandardScaler()
                df[chans] = scaler.fit_transform(df[chans])
                
                # D. Fenêtrage
                for seg, label in pp.get_windows(df):
                    X_list.append(seg)
                    y_list.append(label)
                    sub_list.append(subject_id)
            except: pass
        print(f"   -> Sujet {subject_id} traité")

    if not X_list:
        print("ERREUR CRITIQUE: Aucune donnée trouvée.")
        sys.exit()

    if not os.path.exists(cfg.TENSOR_DIR): os.makedirs(cfg.TENSOR_DIR)
    
    np.save(os.path.join(cfg.TENSOR_DIR, 'X.npy'), np.array(X_list, dtype='float32'))
    np.save(os.path.join(cfg.TENSOR_DIR, 'y.npy'), np.array(y_list, dtype='int32'))
    np.save(os.path.join(cfg.TENSOR_DIR, 'sub.npy'), np.array(sub_list, dtype='int32'))
    print("   [DL] Fichiers .npy sauvegardés.")

if __name__ == "__main__":
    generate_tensors()