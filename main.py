import common.config as cfg

from ml_approach.train_ml import run_ml_experiment
from dl_approaches.CNN.train_cnn import run_cnn_experiment
from dl_approaches.VIT.train_vit import run_vit_experiment
from dl_approaches.GNN.train_gnn import run_gnn_experiment

def main():
    print("=" * 60)
    print("sEMG PROJECT LAUNCH")
    print(f"CURRENT APPROACH : {cfg.MODE}")
    print("=" * 60)

    if cfg.MODE == "ML":
        print(">> Running Feature Engineering + SVM approach...")
        run_ml_experiment()
        
    elif cfg.MODE == "CNN":
        print(">> Running Convolutional Neural Network (CNN) approach...")
        run_cnn_experiment(cfg.MODE)

    elif cfg.MODE == "CNN-no-preprocessing":
        print(">> Running Convolutional Neural Network (CNN) approach without preprocessing data...")
        run_cnn_experiment(cfg.MODE)

    elif cfg.MODE == "CNN-VGG-transfert":
        print(">> Running Convolutional Neural Network (CNN) approach with VGG Transfert Learning model...")
        run_cnn_experiment(cfg.MODE)
    
    elif cfg.MODE == "VIT":
        print(">> Running Vision Transformer (ViT) approach...")
        run_vit_experiment()
  
    elif cfg.MODE == "GNN":
        print(">> Running Graph Neural Network (GNN) approach...")
        run_gnn_experiment()
    
    else:
        print(f"ERROR: The mode '{cfg.MODE}' defined in common/config.py is unknown.")
        print("Please use 'ML', 'CNN', 'VIT', or 'GNN'.")

if __name__ == "__main__":
    main()
