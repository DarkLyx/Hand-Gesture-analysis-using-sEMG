import sys
import os
import shutil
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import signal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import common.config as cfg
import common.preprocessing as pp

def generate_tensors():
    X_list, y_list, sub_list = [], [], []
    print("[DL] Tensors generation (Filters & Z-Score)")
    if os.path.exists(cfg.TENSOR_DIR):
        shutil.rmtree(cfg.TENSOR_DIR)
        
    for subject_id in range(1, 37):
        path = os.path.join(cfg.ROOT_DIR, f"{subject_id:02d}")
        
        for f in glob.glob(os.path.join(path, "*.txt")):
            try:
                df = pd.read_csv(f, sep='\t')
                chans = [c for c in df.columns if 'channel' in c]
                
                if cfg.USE_PREPROCESSING:
                    # Filtres (Notch + Bandpass)
                    df[chans] = pp.butter_bandpass_filter(pp.notch_filter(df[chans].values))

                    # Cleaning
                    df = pp.clean_and_trim_data(df)
                    if df is None: continue
                
                # Z-SCORE
                scaler = StandardScaler()
                df[chans] = scaler.fit_transform(df[chans])
                
                # Windowing
                for seg, label in pp.get_windows(df):
                    if cfg.MODE == "CNN-VGG-transfert":
                        spectros = []
                        for i in range(8):
                            f, t, Sxx = signal.spectrogram(seg[:, i], fs=cfg.FS, nperseg=cfg.SPECTRO_NPERSEG, noverlap=cfg.SPECTRO_NOVERLAP)
                            Sxx = 10 * np.log10(Sxx + 1e-10) 
                            spectros.append(Sxx)
                        
                        full_image = np.vstack(spectros)
                        
                        full_image = np.expand_dims(full_image, axis=-1)
                        
                        X_list.append(full_image)
                    else:
                        X_list.append(seg)
                    y_list.append(label)
                    sub_list.append(subject_id)
                
            except: pass
        print(f"-> Subject {subject_id} done")

    if not X_list:
        print("No data found.")
        sys.exit()

    if not os.path.exists(cfg.TENSOR_DIR): os.makedirs(cfg.TENSOR_DIR)
    
    np.save(os.path.join(cfg.TENSOR_DIR, 'X.npy'), np.array(X_list, dtype='float32'))
    np.save(os.path.join(cfg.TENSOR_DIR, 'y.npy'), np.array(y_list, dtype='int32'))
    np.save(os.path.join(cfg.TENSOR_DIR, 'sub.npy'), np.array(sub_list, dtype='int32'))
    print("[DL] files .npy saved.")

if __name__ == "__main__":
    generate_tensors()