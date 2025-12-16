import common.config as cfg

from ml_approach.train_ml import run_ml_experiment
from dl_approaches.train_cnn import run_cnn_experiment
from dl_approaches.train_vit import run_vit_experiment
from dl_approaches.train_gnn import run_gnn_experiment

def main():
    print("="*60)
    print(f"   LANCEMENT DU PROJET sEMG")
    print(f"   MODE ACTUEL : {cfg.MODE}")
    print("="*60)

    if cfg.MODE == "ML":
        print(">> Lancement de l'approche Feature Engineering + SVM...")
        run_ml_experiment()
        
    elif cfg.MODE == "CNN":
        print(">> Lancement de l'approche Convolutional neural networks (CNN)...")
        run_cnn_experiment()
    
    elif cfg.MODE == "VIT":
        print(">> Lancement de l'approche Vision Transformer (ViT)...")
        run_vit_experiment()
  
    elif cfg.MODE == "GNN":
        print(">> Lancement de l'approche Graph Neural Network (GNN)...")
        run_gnn_experiment()
    
    else:
        print(f"ERREUR : Le mode '{cfg.MODE}' défini dans common/config.py est inconnu.")
        print("Utilisez 'ML', 'CNN' ou 'VIT'.")

if __name__ == "__main__":
    main()