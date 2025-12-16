import os
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from scipy.stats import mode
import common.config as cfg

if not os.path.exists(cfg.RESULTS_DIR): os.makedirs(cfg.RESULTS_DIR)

def majority_voting(predictions, window_size):
    if window_size <= 1: return predictions
    smoothed = []
    for i in range(len(predictions)):
        start = max(0, i - window_size + 1)
        m = mode(predictions[start : i+1], keepdims=True).mode[0]
        smoothed.append(m)
    return np.array(smoothed)

def save_plot(filename):
    path = os.path.join(cfg.RESULTS_DIR, filename)
    plt.savefig(path)
    print(f"   [PLOT] Sauvegardé : {path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, mode_name="ML", dataset_name="Test"):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    nb_classes = cm.shape[0]
    labels = [str(i) for i in range(1, nb_classes + 1)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"{mode_name} - {dataset_name}: Matrice de Confusion")
    plt.ylabel('Vrai Geste')
    plt.xlabel('Geste Prédit')
    save_plot(f"{mode_name}_{dataset_name}_confusion_matrix.png")

def plot_training_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Historique : Précision (Accuracy)')
    plt.xlabel('Époques')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("DL_history_accuracy.png")

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss', color='orange')
    plt.plot(history.history['val_loss'], label='Val Loss', color='red')
    plt.title('Historique : Perte (Loss)')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("DL_history_loss.png")

def save_comparison_results(y_val_true, y_val_pred, y_test_true, y_test_pred, mode_name="ML"):
    """
    Crée un graphique comparatif : Validation vs Test
    pour Accuracy, F1-Macro et F1-Weighted.
    """
    acc_val = accuracy_score(y_val_true, y_val_pred)
    f1m_val = f1_score(y_val_true, y_val_pred, average='macro')
    f1w_val = f1_score(y_val_true, y_val_pred, average='weighted')

    acc_test = accuracy_score(y_test_true, y_test_pred)
    f1m_test = f1_score(y_test_true, y_test_pred, average='macro')
    f1w_test = f1_score(y_test_true, y_test_pred, average='weighted')

    print(f"\n   >>> RÉSULTATS COMPARATIFS ({mode_name}) <<<")
    print(f"   [Validation] Acc: {acc_val:.2%} | F1-Macro: {f1m_val:.2%} | F1-Weighted: {f1w_val:.2%}")
    print(f"   [Test Final] Acc: {acc_test:.2%} | F1-Macro: {f1m_test:.2%} | F1-Weighted: {f1w_test:.2%}")

    metrics = ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)']
    val_scores = [acc_val, f1m_val, f1w_val]
    test_scores = [acc_test, f1m_test, f1w_test]

    x = np.arange(len(metrics)) 
    width = 0.35 

    plt.figure(figsize=(10, 6))
    
    bars1 = plt.bar(x - width/2, val_scores, width, label='Validation', color='#FF9800', alpha=0.8)
    bars2 = plt.bar(x + width/2, test_scores, width, label='Test', color='#4CAF50', alpha=0.8)

    plt.ylabel('Score (0-1)')
    plt.title(f'{mode_name}: Comparaison Validation vs Test')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.15)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.1%}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    add_labels(bars1)
    add_labels(bars2)

    save_plot(f"{mode_name}_comparison_metrics.png")