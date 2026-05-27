"""
preprocess.py — Signal preprocessing pipeline.

Steps:
  1. Bandpass filter  : Butterworth 4th-order, 0.5–40 Hz
  2. Sliding window   : 200-sample windows, 100-sample step (50% overlap)
  3. Feature extract  : Per-channel [mean, std, rms, zcr, dominant_freq] → RF
  4. Train/val/test split by subject
  5. Z-score normalization (train stats only)
  6. Save norm stats to config/
"""

import os
import numpy as np
from pathlib import Path

try:
    from scipy.signal import butter, sosfilt
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[preprocess] WARNING: scipy not available — bandpass filter disabled")


# ── Filtering ───────────────────────────────────────────────────────────────

def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = 1000.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply 4th-order Butterworth bandpass filter to every channel.

    Args:
        signal : (n_samples, n_channels) float32
        lowcut : lower cutoff frequency (Hz)
        highcut: upper cutoff frequency (Hz)
        fs     : sampling frequency (Hz)
        order  : filter order

    Returns:
        Filtered signal, same shape as input.
    """
    if not SCIPY_OK:
        return signal  # graceful degradation

    nyq = fs / 2.0
    lo = lowcut / nyq
    hi = highcut / nyq

    # Clamp to valid range for butter
    lo = max(lo, 1e-6)
    hi = min(hi, 1.0 - 1e-6)

    sos = butter(order, [lo, hi], btype='band', output='sos')
    return sosfilt(sos, signal.astype(np.float64), axis=0).astype(np.float32)


# ── Windowing ───────────────────────────────────────────────────────────────

def sliding_window(
    signal: np.ndarray,
    window_size: int = 200,
    step: int = 100,
) -> np.ndarray:
    """
    Segment a signal into overlapping windows.

    Args:
        signal     : (n_samples, n_channels)
        window_size: samples per window (200 @ 1 kHz = 200 ms)
        step       : hop size (100 → 50 % overlap)

    Returns:
        windows: (N_windows, window_size, n_channels)
    """
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    n_samples, n_channels = signal.shape
    windows = []

    for start in range(0, n_samples - window_size + 1, step):
        windows.append(signal[start: start + window_size, :])

    if not windows:
        return np.empty((0, window_size, n_channels), dtype=np.float32)

    return np.stack(windows, axis=0).astype(np.float32)


# ── Feature extraction (for RF) ─────────────────────────────────────────────

def _zero_crossing_rate(x: np.ndarray) -> float:
    """Fraction of samples where sign changes."""
    return float(np.sum(np.diff(np.sign(x)) != 0)) / len(x)


def _dominant_frequency(x: np.ndarray, fs: float = 1000.0) -> float:
    """Frequency (Hz) of the highest-amplitude spectral bin (excluding DC)."""
    fft_mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    fft_mag[0] = 0.0  # zero out DC
    idx = np.argmax(fft_mag)
    return float(freqs[idx])


def extract_features(windows: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """
    Extract handcrafted time- and frequency-domain features per window.

    Per channel extracts: [mean, std, rms, zero_crossing_rate, dominant_freq_hz]

    Args:
        windows : (N_windows, window_size, n_channels)
        fs      : sampling frequency

    Returns:
        features: (N_windows, n_channels * 5)
    """
    n_windows, window_size, n_channels = windows.shape
    features = np.zeros((n_windows, n_channels * 5), dtype=np.float32)

    for wi in range(n_windows):
        w = windows[wi]  # (window_size, n_channels)
        for ch in range(n_channels):
            x = w[:, ch].astype(np.float64)
            base = ch * 5
            features[wi, base + 0] = float(np.mean(x))
            features[wi, base + 1] = float(np.std(x))
            features[wi, base + 2] = float(np.sqrt(np.mean(x ** 2)))
            features[wi, base + 3] = _zero_crossing_rate(x)
            features[wi, base + 4] = _dominant_frequency(x, fs)

    return features


# ── Subject-independent train/val/test split ────────────────────────────────

def train_test_split_by_subject(
    X_list: list,
    y_list: list,
    subjects: list,
    train_subj=None,
    val_subj=None,
    test_subj=None,
    window_size: int = 200,
    step: int = 100,
    fs: float = 1000.0,
    config_dir: str = None,
) -> tuple:
    """
    Subject-independent split: different subjects in train / val / test.

    Default split (10 subjects):
      train : subjects 1–7
      val   : subjects 8–9
      test  : subject 10

    For synthetic data (subject IDs 1–10 each have all 7 classes) the
    same IDs are used — each call with the same subject list is reproducible.

    Args:
        X_list      : list of (n_samples, n_channels) arrays
        y_list      : list of int labels
        subjects    : list of subject IDs
        train_subj  : list of subject IDs for training
        val_subj    : list of subject IDs for validation
        test_subj   : list of subject IDs for test
        window_size : window length in samples
        step        : hop size
        fs          : sampling frequency
        config_dir  : where to save norm_mean.npy / norm_std.npy
                      (default: config/ relative to cwd)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
        (arrays; windows shape = (N, window_size, n_channels))
    """
    unique_subjects = sorted(set(subjects))
    n_subj = len(unique_subjects)

    if train_subj is None:
        # 70/15/15 split by subject count
        n_train = max(1, int(n_subj * 0.70))
        n_val   = max(1, int(round(n_subj * 0.15)))
        n_val   = max(n_val, 1)
        # Ensure we have at least 1 test subject
        remaining = n_subj - n_train - n_val
        if remaining < 1 and n_subj >= 3:
            n_val -= 1
        train_subj = unique_subjects[:n_train]
        val_subj   = unique_subjects[n_train: n_train + n_val]
        test_subj  = unique_subjects[n_train + n_val:]
        if not test_subj:
            test_subj = [unique_subjects[-1]]
        print(f"[preprocess] Auto split — train: {train_subj}, "
              f"val: {val_subj}, test: {test_subj}")

    def _process_subset(subset_subjects):
        X_wins, y_wins = [], []
        for sig, label, subj in zip(X_list, y_list, subjects):
            if subj not in subset_subjects:
                continue
            filtered = bandpass_filter(sig, fs=fs)
            wins = sliding_window(filtered, window_size, step)
            if wins.shape[0] == 0:
                continue
            X_wins.append(wins)
            y_wins.extend([label] * wins.shape[0])
        if not X_wins:
            n_ch = X_list[0].shape[1] if X_list else 1
            return (np.empty((0, window_size, n_ch), dtype=np.float32),
                    np.array([], dtype=np.int64))
        return (np.concatenate(X_wins, axis=0),
                np.array(y_wins, dtype=np.int64))

    X_train, y_train = _process_subset(train_subj)
    X_val,   y_val   = _process_subset(val_subj)
    X_test,  y_test  = _process_subset(test_subj)

    # Compute normalization statistics on training set
    if X_train.shape[0] > 0:
        n_ch = X_train.shape[2]
        flat = X_train.reshape(-1, n_ch)
        norm_mean = flat.mean(axis=0).astype(np.float32)
        norm_std  = (flat.std(axis=0) + 1e-8).astype(np.float32)
    else:
        n_ch = X_list[0].shape[1] if X_list else 1
        norm_mean = np.zeros(n_ch, dtype=np.float32)
        norm_std  = np.ones(n_ch, dtype=np.float32)

    def _normalize(X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return X
        sh = X.shape
        flat = X.reshape(-1, n_ch)
        return ((flat - norm_mean) / norm_std).reshape(sh).astype(np.float32)

    X_train = _normalize(X_train)
    X_val   = _normalize(X_val)
    X_test  = _normalize(X_test)

    # Save normalization stats
    if config_dir is None:
        config_dir = 'config'
    os.makedirs(config_dir, exist_ok=True)
    np.save(os.path.join(config_dir, 'norm_mean.npy'), norm_mean)
    np.save(os.path.join(config_dir, 'norm_std.npy'),  norm_std)
    print(f"[preprocess] Norm stats saved to {config_dir}/")

    print(f"[preprocess] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from gait_intent.data_loader import load_dataset

    print("Loading data …")
    X_list, y_list, subjects, synthetic = load_dataset()

    print("Splitting …")
    X_tr, y_tr, X_v, y_v, X_te, y_te = train_test_split_by_subject(
        X_list, y_list, subjects
    )

    print("Extracting features for RF …")
    F_tr = extract_features(X_tr[:500])  # sample
    print(f"Feature matrix shape: {F_tr.shape}")

    print("Done.")
