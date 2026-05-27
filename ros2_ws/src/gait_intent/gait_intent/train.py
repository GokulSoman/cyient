"""
train.py — Training loop for gait intent recognition models.

Pipeline:
  1. Load data (real .mat or synthetic fallback)
  2. Preprocess: filter, window, split, normalize
  3. Train Random Forest on hand-crafted features
  4. Train 1D CNN with Adam + ReduceLROnPlateau + EarlyStopping
  5. Save best models to results/

Usage:
  python3 train.py [--data-dir <path>] [--epochs <N>] [--no-cnn]
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
from pathlib import Path

# ── Resolve package path ─────────────────────────────────────────────────────
_PKG = Path(__file__).resolve().parents[2]  # …/ros2_ws/src/gait_intent/
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from gait_intent.data_loader  import load_dataset, CLASS_NAMES, N_CLASSES
from gait_intent.preprocess   import train_test_split_by_subject, extract_features
from gait_intent.classifier   import build_random_forest, TORCH_OK

REPO_ROOT = Path(__file__).resolve().parents[4]  # cyient/
RESULTS_DIR = REPO_ROOT / 'results'
CONFIG_DIR  = REPO_ROOT / 'ros2_ws' / 'src' / 'gait_intent' / 'config'


def train_random_forest(X_train_feat, y_train, X_val_feat, y_val):
    """Fit RF on feature matrix; report val accuracy."""
    from sklearn.metrics import accuracy_score

    print("\n[train] ── Random Forest ──────────────────────────────────")
    rf = build_random_forest()
    t0 = time.time()
    rf.fit(X_train_feat, y_train)
    elapsed = time.time() - t0

    val_preds = rf.predict(X_val_feat)
    val_acc   = accuracy_score(y_val, val_preds)
    print(f"  Training time : {elapsed:.1f} s")
    print(f"  Val accuracy  : {val_acc * 100:.2f} %")
    return rf, val_acc


def train_cnn(X_train, y_train, X_val, y_val,
              n_classes, epochs=50, batch_size=64, patience=10,
              results_dir=None):
    """Train GaitCNN1D with Adam + LR scheduling + early stopping."""
    if not TORCH_OK:
        print("[train] PyTorch not available — skipping CNN training.")
        return None

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from gait_intent.classifier import build_cnn

    if results_dir is None:
        results_dir = Path('results')
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    n_channels  = X_train.shape[2]
    window_size = X_train.shape[1]

    # Build model + device
    model, device = build_cnn(n_channels=n_channels, n_classes=n_classes,
                               window_size=window_size)
    print(f"\n[train] ── 1D CNN ─────────────────────────────────────────")
    print(f"  Architecture  : GaitCNN1D({n_channels}ch → {n_classes}cls)")
    print(f"  Parameters    : {model.count_parameters():,}")
    print(f"  Device        : {device}")

    # DataLoaders
    class AugmentedDataset(torch.utils.data.Dataset):
        """Simple training augmentation: Gaussian noise + amplitude scaling."""
        def __init__(self, X, y, augment=False):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y.astype(np.int64))
            self.augment = augment

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            x = self.X[idx].clone()
            if self.augment:
                # Random Gaussian noise
                x = x + 0.05 * torch.randn_like(x)
                # Random amplitude scale [0.85, 1.15]
                scale = 0.85 + 0.30 * torch.rand(1).item()
                x = x * scale
            return x, self.y[idx]

    def _loader(X, y, shuffle, augment=False):
        ds = AugmentedDataset(X, y, augment=augment)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(device.type == 'cuda'))

    train_loader = _loader(X_train, y_train, shuffle=True, augment=True)
    val_loader   = _loader(X_val,   y_val,   shuffle=False, augment=False)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler  = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                   patience=5)

    best_val_loss = 0.0   # tracking best val accuracy (higher is better)
    best_epoch    = 0
    best_path     = results_dir / 'gait_cnn.pt'
    train_losses, val_losses, val_accs = [], [], []
    no_improve = 0

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        ep_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(yb)
        ep_loss /= len(y_train)

        # ── Validate ──
        model.eval()
        v_loss, correct = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                v_loss += criterion(logits, yb).item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
        v_loss /= len(y_val)
        v_acc   = correct / len(y_val)

        train_losses.append(ep_loss)
        val_losses.append(v_loss)
        val_accs.append(v_acc)
        scheduler.step(v_acc)

        # ── Early stopping (track best val accuracy) ──
        if v_acc > best_val_loss:  # reusing variable for val accuracy
            best_val_loss = v_acc
            best_epoch    = epoch
            no_improve    = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={ep_loss:.4f}  val_loss={v_loss:.4f}  "
                  f"val_acc={v_acc*100:.1f}%  lr={lr_now:.2e}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    total_time = time.time() - t_start
    print(f"  Training time : {total_time:.1f} s  |  best epoch : {best_epoch}  "
          f"|  best val acc : {best_val_loss*100:.2f}%")
    print(f"  Best model saved → {best_path}")

    # Reload best weights
    model.load_state_dict(torch.load(best_path, map_location=device,
                                     weights_only=True))
    model.eval()

    # Save training curves
    _save_training_curves(train_losses, val_losses, val_accs, results_dir)

    return model


def _save_training_curves(train_losses, val_losses, val_accs, results_dir):
    """Save training/validation loss and accuracy curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plots_dir = Path(results_dir) / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(train_losses, label='train loss')
        axes[0].plot(val_losses,   label='val loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot([a * 100 for a in val_accs], color='green', label='val acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        out_path = plots_dir / 'training_curves.png'
        plt.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Training curves → {out_path}")
    except Exception as e:
        print(f"  [train] Could not save training curves: {e}")


def run_training(
    data_dir: str = None,
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 10,
    train_cnn_flag: bool = True,
    results_dir: str = None,
    config_dir: str = None,
):
    """
    Full training pipeline: load → preprocess → train RF + CNN → save.

    Returns:
        rf_model  : fitted RandomForestClassifier
        cnn_model : trained GaitCNN1D (or None if skipped / torch missing)
        split_data: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    if data_dir is None:
        data_dir = str(REPO_ROOT / 'data' / 'gait')
    if results_dir is None:
        results_dir = str(RESULTS_DIR)
    if config_dir is None:
        config_dir = str(CONFIG_DIR)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(config_dir,  exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("[train] Step 1: Load dataset")
    print("=" * 60)
    X_list, y_list, subjects, synthetic = load_dataset(data_dir=data_dir)
    if len(X_list) == 0:
        print("[train] ERROR: No data loaded. Exiting.")
        return None, None, None

    # ── 2. Preprocess / split ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[train] Step 2: Preprocess (filter → window → normalize → split)")
    print("=" * 60)
    X_tr, y_tr, X_v, y_v, X_te, y_te = train_test_split_by_subject(
        X_list, y_list, subjects,
        config_dir=config_dir,
    )

    # ── 3. Random Forest ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[train] Step 3: Random Forest")
    print("=" * 60)
    print("  Extracting hand-crafted features …")
    F_tr = extract_features(X_tr)
    F_v  = extract_features(X_v)
    F_te = extract_features(X_te)

    rf, rf_val_acc = train_random_forest(F_tr, y_tr, F_v, y_v)

    rf_path = os.path.join(results_dir, 'gait_rf.pkl')
    with open(rf_path, 'wb') as fh:
        pickle.dump(rf, fh)
    print(f"  RF saved → {rf_path}")

    # ── 4. 1D CNN ─────────────────────────────────────────────────────────────
    cnn_model = None
    if train_cnn_flag:
        print("\n" + "=" * 60)
        print("[train] Step 4: 1D CNN")
        print("=" * 60)
        cnn_model = train_cnn(
            X_tr, y_tr, X_v, y_v,
            n_classes=N_CLASSES,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            results_dir=results_dir,
        )

    print("\n[train] ── Training complete ──────────────────────────────")
    return rf, cnn_model, (X_tr, y_tr, X_v, y_v, X_te, y_te)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train gait intent models')
    parser.add_argument('--data-dir', default=None,
                        help='Path to data/gait/ (default: auto-detect)')
    parser.add_argument('--epochs',   type=int, default=50)
    parser.add_argument('--batch',    type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--no-cnn',   action='store_true',
                        help='Skip CNN, train RF only')
    args = parser.parse_args()

    run_training(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
        train_cnn_flag=not args.no_cnn,
    )
