#!/usr/bin/env python3
"""
run_pipeline.py — Single-script end-to-end demo of the gait intent pipeline.

Stages:
  1. Load / generate data
  2. Preprocess (filter → window → normalize → split)
  3. Train Random Forest + 1D CNN (skip if saved models exist)
  4. Evaluate both models (accuracy, F1, confusion matrix)
  5. Run FSM demo (state-transition table)
  6. Print final summary

Usage:
  python3 run_pipeline.py [--retrain] [--no-cnn] [--epochs N]

Options:
  --retrain   Force retraining even if saved models exist
  --no-cnn    Skip CNN training (faster, RF only)
  --epochs N  Number of CNN training epochs (default: 50)
  --data-dir  Path to real data (default: auto-detect repo/data/gait)
"""

import os
import sys
import time
import argparse
from pathlib import Path

# ── Resolve package paths ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent        # scripts/
_PKG_DIR    = _SCRIPT_DIR.parent                     # gait_intent/ (ROS pkg root)
_REPO_ROOT  = _PKG_DIR.parents[2]                    # cyient/

# Add the ros2_ws/src/gait_intent/ directory to sys.path
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

RESULTS_DIR = _REPO_ROOT / 'results'
CONFIG_DIR  = _PKG_DIR   / 'config'
DATA_DIR    = _REPO_ROOT / 'data' / 'gait'

os.makedirs(str(RESULTS_DIR / 'plots'), exist_ok=True)
os.makedirs(str(CONFIG_DIR), exist_ok=True)
os.makedirs(str(DATA_DIR),   exist_ok=True)


# ── Imports ───────────────────────────────────────────────────────────────────
print("[pipeline] Importing modules …")
from gait_intent.data_loader  import load_dataset, CLASS_NAMES, N_CLASSES
from gait_intent.preprocess   import train_test_split_by_subject, extract_features
from gait_intent.classifier   import build_random_forest, TORCH_OK
from gait_intent.control_fsm  import ExoskeletonFSM, PREDICTION_TO_STATE, STATES, demo as fsm_demo


def banner(title: str) -> None:
    width = 60
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def run_pipeline(
    data_dir: str = None,
    retrain: bool = False,
    train_cnn_flag: bool = True,
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 10,
) -> None:
    t_total = time.time()

    if data_dir is None:
        data_dir = str(DATA_DIR)

    rf_path  = str(RESULTS_DIR / 'gait_rf.pkl')
    cnn_path = str(RESULTS_DIR / 'gait_cnn.pt')

    # ── Stage 1: Load data ────────────────────────────────────────────────────
    banner("STAGE 1 — Data Loading")
    X_list, y_list, subjects, synthetic = load_dataset(data_dir=data_dir)

    from collections import Counter
    class_counts = Counter(y_list)
    print(f"\nDataset info:")
    print(f"  Source     : {'SYNTHETIC' if synthetic else 'REAL .mat files'}")
    print(f"  Segments   : {len(X_list)}")
    print(f"  Subjects   : {sorted(set(subjects))}")
    print(f"  Signal dim : {X_list[0].shape}")
    print(f"  Class distribution:")
    for cls_idx in sorted(class_counts):
        print(f"    [{cls_idx}] {CLASS_NAMES[cls_idx]:20s}: {class_counts[cls_idx]} segs")

    # ── Stage 2: Preprocess ───────────────────────────────────────────────────
    banner("STAGE 2 — Preprocessing")
    X_tr, y_tr, X_v, y_v, X_te, y_te = train_test_split_by_subject(
        X_list, y_list, subjects,
        config_dir=str(CONFIG_DIR),
    )
    print(f"\n  Window shape : {X_tr.shape[1:]} (window_size × n_channels)")
    print(f"  Train windows: {len(y_tr)}")
    print(f"  Val windows  : {len(y_v)}")
    print(f"  Test windows : {len(y_te)}")

    # ── Stage 3: Train ────────────────────────────────────────────────────────
    banner("STAGE 3 — Model Training")
    needs_rf_train  = retrain or not os.path.exists(rf_path)
    needs_cnn_train = train_cnn_flag and (retrain or not os.path.exists(cnn_path))

    rf  = None
    cnn = None

    # Random Forest
    if needs_rf_train:
        import pickle
        from sklearn.metrics import accuracy_score
        print("\n[RF] Extracting features …")
        F_tr = extract_features(X_tr)
        F_v  = extract_features(X_v)

        print("[RF] Fitting …")
        t0 = time.time()
        rf = build_random_forest()
        rf.fit(F_tr, y_tr)
        elapsed = time.time() - t0

        v_acc = accuracy_score(y_v, rf.predict(F_v))
        print(f"[RF] Val accuracy : {v_acc*100:.2f}%  ({elapsed:.1f} s)")
        with open(rf_path, 'wb') as fh:
            pickle.dump(rf, fh)
        print(f"[RF] Saved → {rf_path}")
    else:
        print(f"[RF] Using cached model: {rf_path}")

    # 1D CNN
    if needs_cnn_train and TORCH_OK:
        print()
        from gait_intent.train import train_cnn
        cnn = train_cnn(
            X_tr, y_tr, X_v, y_v,
            n_classes=N_CLASSES,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            results_dir=str(RESULTS_DIR),
        )
    elif not TORCH_OK:
        print("[CNN] PyTorch not available — skipping CNN.")
    else:
        print(f"[CNN] Using cached model: {cnn_path}")

    # ── Stage 4: Evaluate ─────────────────────────────────────────────────────
    banner("STAGE 4 — Evaluation")
    from gait_intent.evaluate import run_evaluation
    results = run_evaluation(
        split_data=(X_tr, y_tr, X_v, y_v, X_te, y_te),
        rf_path=rf_path,
        cnn_path=cnn_path,
        results_dir=str(RESULTS_DIR),
        config_dir=str(CONFIG_DIR),
    )

    # ── Stage 5: FSM demo ─────────────────────────────────────────────────────
    banner("STAGE 5 — Assistive Control FSM Demo")
    fsm_demo(verbose=True)

    # ── Stage 6: Summary ──────────────────────────────────────────────────────
    banner("STAGE 6 — Final Summary")

    elapsed_total = time.time() - t_total
    print(f"\n  Pipeline completed in {elapsed_total:.1f} s")
    print(f"\n  Data source : {'synthetic' if synthetic else 'real'}")
    print(f"  Train/Val/Test: {len(y_tr)} / {len(y_v)} / {len(y_te)} windows")

    if results:
        print(f"\n  Model Performance:")
        for name, m in results.items():
            print(f"    {name.upper():6s}: accuracy={m['accuracy']*100:.2f}%  "
                  f"macro-F1={m['macro_f1']:.4f}  "
                  f"latency={m.get('latency_mean_ms', 0):.2f}ms")

    print(f"\n  Output files:")
    plots = list(Path(RESULTS_DIR / 'plots').glob('*.png'))
    for p in sorted(plots):
        print(f"    {p}")
    print(f"    {RESULTS_DIR / 'gait_rf.pkl'}")
    if TORCH_OK:
        print(f"    {RESULTS_DIR / 'gait_cnn.pt'}")

    print("\n  FSM torque reference:")
    for name, st in STATES.items():
        print(f"    {name:16s} : torque={st.torque_scale:.1f}")

    target_acc = 0.80
    best_acc = max((m['accuracy'] for m in results.values()), default=0.0)
    target_met = "PASS" if best_acc >= target_acc else "BELOW TARGET"
    print(f"\n  Target accuracy (>=80%): {best_acc*100:.2f}% → {target_met}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gait intent end-to-end pipeline')
    parser.add_argument('--data-dir',  default=None)
    parser.add_argument('--retrain',   action='store_true',
                        help='Force retraining even if models exist')
    parser.add_argument('--no-cnn',    action='store_true',
                        help='Skip CNN training')
    parser.add_argument('--epochs',    type=int, default=50)
    parser.add_argument('--batch',     type=int, default=64)
    parser.add_argument('--patience',  type=int, default=10)
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        retrain=args.retrain,
        train_cnn_flag=not args.no_cnn,
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
    )
