"""
evaluate.py — Evaluation: metrics, confusion matrix, latency.

Outputs:
  - Accuracy, macro-F1, per-class precision/recall/F1
  - Confusion matrix → results/plots/confusion_matrix.png
  - Sample prediction plot → results/plots/sample_predictions.png
  - Inference latency (mean ± std over 1000 runs)
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
from pathlib import Path

matplotlib_ok = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib_ok = True
except ImportError:
    pass

# ── Resolve package path ─────────────────────────────────────────────────────
_PKG = Path(__file__).resolve().parents[2]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from gait_intent.data_loader import CLASS_NAMES, N_CLASSES
from gait_intent.classifier  import TORCH_OK

REPO_ROOT   = Path(__file__).resolve().parents[4]
RESULTS_DIR = REPO_ROOT / 'results'
CONFIG_DIR  = REPO_ROOT / 'ros2_ws' / 'src' / 'gait_intent' / 'config'


# ── Metric utilities ─────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute accuracy, macro-F1, and per-class precision/recall/F1."""
    from sklearn.metrics import (
        accuracy_score, f1_score,
        precision_recall_fscore_support, classification_report
    )

    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec, rec, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0
    )

    return {
        'accuracy'   : acc,
        'macro_f1'   : f1,
        'precision'  : prec,
        'recall'     : rec,
        'f1_per_class': f1_per,
        'support'    : support,
        'report'     : classification_report(
            y_true, y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        ),
    }


def print_metrics(metrics: dict, model_name: str = '') -> None:
    """Pretty-print evaluation metrics."""
    tag = f" ({model_name})" if model_name else ""
    print(f"\n{'=' * 58}")
    print(f" Evaluation Results{tag}")
    print(f"{'=' * 58}")
    print(f"  Accuracy  : {metrics['accuracy']*100:.2f} %")
    print(f"  Macro F1  : {metrics['macro_f1']:.4f}")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':20s} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
    print(f"  {'-' * 46}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:20s} "
              f"{metrics['precision'][i]:6.3f} "
              f"{metrics['recall'][i]:6.3f} "
              f"{metrics['f1_per_class'][i]:6.3f} "
              f"{metrics['support'][i]:6d}")
    print()


# ── Confusion matrix ─────────────────────────────────────────────────────────

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: str = 'Confusion Matrix',
) -> None:
    """Compute and save a normalised confusion matrix figure."""
    if not matplotlib_ok:
        print("[evaluate] matplotlib not available — confusion matrix not saved")
        return

    from sklearn.metrics import confusion_matrix

    cm  = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    short_names = [n.replace('_', '\n') for n in CLASS_NAMES]
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)

    thresh = 0.5
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            color = 'white' if cm_norm[i, j] > thresh else 'black'
            ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                    ha='center', va='center', fontsize=7, color=color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix → {out_path}")


# ── Sample prediction plot ────────────────────────────────────────────────────

def save_sample_predictions(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    n_windows: int = 40,
) -> None:
    """Plot ground-truth vs predicted labels for a sample of test windows."""
    if not matplotlib_ok:
        return

    n_windows = min(n_windows, len(y_test))
    indices   = np.arange(n_windows)

    correct = (y_pred[:n_windows] == y_test[:n_windows]).astype(int)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8))

    # Row 0: raw signal (channel 2 — vertical accel)
    if X_test.shape[0] > 0 and X_test.shape[2] > 2:
        sig = X_test[:n_windows, :, 2].reshape(-1)
        axes[0].plot(sig, lw=0.8, color='steelblue')
        axes[0].set_title('Test signal — vertical accelerometer (ch 2, normalised)')
        axes[0].set_ylabel('Amplitude (norm.)')
        axes[0].set_xlabel('Sample')
        axes[0].grid(True, alpha=0.3)

    # Row 1: true vs predicted class
    axes[1].step(indices, y_test[:n_windows],  where='post', label='True',
                 color='green',  lw=1.5)
    axes[1].step(indices, y_pred[:n_windows],  where='post', label='Pred',
                 color='red', lw=1.5, linestyle='--')
    axes[1].set_yticks(range(N_CLASSES))
    axes[1].set_yticklabels(CLASS_NAMES, fontsize=7)
    axes[1].set_title('True vs Predicted class labels')
    axes[1].set_xlabel('Window index')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Row 2: correct/incorrect
    colors = ['green' if c else 'red' for c in correct]
    axes[2].bar(indices, correct, color=colors, width=0.8)
    axes[2].set_ylim(-0.1, 1.2)
    axes[2].set_title('Correct (green) / Wrong (red) predictions')
    axes[2].set_xlabel('Window index')
    axes[2].set_ylabel('Correct?')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Sample Predictions on Test Set', fontsize=12, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Sample predictions → {out_path}")


# ── Latency measurement ───────────────────────────────────────────────────────

def measure_rf_latency(rf, X_feat: np.ndarray, n_runs: int = 1000) -> tuple:
    """Time per-sample RF inference."""
    single_feat = X_feat[:1]
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        rf.predict(single_feat)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times[100:])  # discard warm-up
    return times.mean(), times.std()


def measure_cnn_latency(model, n_channels: int,
                         window_size: int = 200, n_runs: int = 1000) -> tuple:
    """Time per-window CNN inference on CPU."""
    if not TORCH_OK:
        return 0.0, 0.0
    import torch
    model.eval()
    dummy = torch.randn(1, window_size, n_channels)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times[100:])
    return times.mean(), times.std()


# ── Main evaluation function ──────────────────────────────────────────────────

def run_evaluation(
    split_data=None,
    rf_path: str = None,
    cnn_path: str = None,
    results_dir: str = None,
    config_dir: str = None,
    data_dir: str = None,
) -> dict:
    """
    Load saved models + (optionally pre-split) test data, run full evaluation.

    Args:
        split_data : pre-computed (X_tr, y_tr, X_v, y_v, X_te, y_te) or None
        rf_path    : path to gait_rf.pkl
        cnn_path   : path to gait_cnn.pt
        results_dir: where to write plots
        config_dir : where norm_mean/std are stored

    Returns:
        dict with RF and/or CNN metrics
    """
    from gait_intent.preprocess import extract_features

    if results_dir is None:
        results_dir = str(RESULTS_DIR)
    if rf_path is None:
        rf_path = str(RESULTS_DIR / 'gait_rf.pkl')
    if cnn_path is None:
        cnn_path = str(RESULTS_DIR / 'gait_cnn.pt')
    if config_dir is None:
        config_dir = str(CONFIG_DIR)
    if data_dir is None:
        data_dir = str(REPO_ROOT / 'data' / 'gait')

    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # ── Load test split ─────────────────────────────────────────────────────
    if split_data is None:
        print("[evaluate] No split_data provided — regenerating …")
        from gait_intent.data_loader import load_dataset
        from gait_intent.preprocess  import train_test_split_by_subject

        X_list, y_list, subjects, _ = load_dataset(data_dir=data_dir)
        _, _, _, _, X_te, y_te = train_test_split_by_subject(
            X_list, y_list, subjects,
            config_dir=config_dir,
        )
    else:
        _, _, _, _, X_te, y_te = split_data

    if len(y_te) == 0:
        print("[evaluate] WARNING: Test set is empty!")
        return {}

    print(f"\n[evaluate] Test set: {X_te.shape}, {len(y_te)} windows")

    all_results = {}

    # ── Random Forest evaluation ─────────────────────────────────────────────
    if os.path.exists(rf_path):
        print(f"\n[evaluate] Loading RF from {rf_path} …")
        with open(rf_path, 'rb') as fh:
            rf = pickle.load(fh)

        F_te     = extract_features(X_te)
        y_pred_rf = rf.predict(F_te)
        rf_metrics = compute_metrics(y_te, y_pred_rf)
        print_metrics(rf_metrics, model_name='Random Forest')

        save_confusion_matrix(
            y_te, y_pred_rf,
            out_path=os.path.join(plots_dir, 'confusion_matrix_rf.png'),
            title='Confusion Matrix — Random Forest',
        )
        save_sample_predictions(
            X_te, y_te, y_pred_rf,
            out_path=os.path.join(plots_dir, 'sample_predictions_rf.png'),
        )

        rf_lat_mean, rf_lat_std = measure_rf_latency(rf, F_te[:200])
        print(f"  RF inference latency: {rf_lat_mean:.3f} ± {rf_lat_std:.3f} ms")

        all_results['rf'] = rf_metrics
        all_results['rf']['latency_mean_ms'] = rf_lat_mean
        all_results['rf']['latency_std_ms']  = rf_lat_std
    else:
        print(f"[evaluate] RF model not found at {rf_path}")

    # ── CNN evaluation ────────────────────────────────────────────────────────
    if TORCH_OK and os.path.exists(cnn_path):
        import torch
        from gait_intent.classifier import GaitCNN1D

        print(f"\n[evaluate] Loading CNN from {cnn_path} …")
        n_channels  = X_te.shape[2]
        window_size = X_te.shape[1]

        model = GaitCNN1D(n_channels=n_channels, n_classes=N_CLASSES,
                           window_size=window_size)
        model.load_state_dict(torch.load(cnn_path, map_location='cpu',
                                          weights_only=True))
        model.eval()

        # Batch inference
        with torch.no_grad():
            logits = model(torch.from_numpy(X_te))
            y_pred_cnn = logits.argmax(dim=1).numpy()

        cnn_metrics = compute_metrics(y_te, y_pred_cnn)
        print_metrics(cnn_metrics, model_name='1D CNN')

        save_confusion_matrix(
            y_te, y_pred_cnn,
            out_path=os.path.join(plots_dir, 'confusion_matrix.png'),
            title='Confusion Matrix — 1D CNN',
        )
        save_sample_predictions(
            X_te, y_te, y_pred_cnn,
            out_path=os.path.join(plots_dir, 'sample_predictions.png'),
        )

        cnn_lat_mean, cnn_lat_std = measure_cnn_latency(
            model, n_channels=n_channels, window_size=window_size
        )
        print(f"  CNN inference latency: {cnn_lat_mean:.3f} ± {cnn_lat_std:.3f} ms")

        all_results['cnn'] = cnn_metrics
        all_results['cnn']['latency_mean_ms'] = cnn_lat_mean
        all_results['cnn']['latency_std_ms']  = cnn_lat_std
    elif not os.path.exists(cnn_path):
        print(f"[evaluate] CNN model not found at {cnn_path} — skipping")

    # ── Summary table ─────────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'=' * 50}")
        print(" SUMMARY")
        print(f"{'=' * 50}")
        print(f"  {'Model':12s} {'Accuracy':>10} {'Macro-F1':>10} {'Latency':>14}")
        print(f"  {'-' * 50}")
        for name, m in all_results.items():
            lat = f"{m.get('latency_mean_ms', 0):.2f} ms"
            print(f"  {name.upper():12s} "
                  f"{m['accuracy']*100:9.2f}%  "
                  f"{m['macro_f1']:10.4f}  "
                  f"{lat:>14}")

    return all_results


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate gait intent models')
    parser.add_argument('--rf',      default=None, help='Path to gait_rf.pkl')
    parser.add_argument('--cnn',     default=None, help='Path to gait_cnn.pt')
    parser.add_argument('--results', default=None, help='Results directory')
    args = parser.parse_args()

    run_evaluation(rf_path=args.rf, cnn_path=args.cnn, results_dir=args.results)
