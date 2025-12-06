import common.config as cfg

from ml_approach.train_ml import run_ml_experiment
from dl_approach.train_cnn import run_dl_experiment

def main():
    print("="*60)
    print(f"   LANCEMENT DU PROJET sEMG")
    print(f"   MODE ACTUEL : {cfg.MODE}")
    print("="*60)

    if cfg.MODE == "ML":
        print(">> Lancement de l'approche Feature Engineering + SVM...")
        run_ml_experiment()
        
    elif cfg.MODE == "DL":
        print(">> Lancement de l'approche Deep Learning (CNN)...")
        run_dl_experiment()
        
    else:
        print(f"ERREUR : Le mode '{cfg.MODE}' défini dans common/config.py est inconnu.")
        print("Utilisez 'ML' ou 'DL'.")

if __name__ == "__main__":
    main()