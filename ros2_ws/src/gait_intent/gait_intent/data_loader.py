"""
data_loader.py — Load HDsEMG+IMU dataset (with synthetic fallback).

Two modes:
  1. Real data mode: load .mat files from data/gait/
  2. Synthetic mode: auto-activated if no real data found.
     Generates realistic IMU time-series for 7 locomotion classes
     with class-specific frequency/amplitude profiles.
"""

import os
import numpy as np
from pathlib import Path

# ── Activity mapping ────────────────────────────────────────────────────────
ACTIVITY_MAP = {
    'level_walking': 0,
    'stair_ascent':  1,
    'stair_descent': 2,
    'ramp_ascent':   3,
    'ramp_descent':  4,
    'standing':      5,
    'sit_to_stand':  6,
}

CLASS_NAMES = [
    'level_walking',
    'stair_ascent',
    'stair_descent',
    'ramp_ascent',
    'ramp_descent',
    'standing',
    'sit_to_stand',
]

N_CLASSES = len(CLASS_NAMES)


# ── Synthetic data generation ───────────────────────────────────────────────

def generate_synthetic_data(
    n_subjects: int = 20,
    n_channels: int = 6,
    fs: float = 1000.0,
    duration_per_activity: float = 30.0,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic IMU data for 7 locomotion modes.

    Each activity has characteristic frequency content:
      0 LEVEL_WALKING : ~1-2 Hz dominant, moderate amplitude
      1 STAIR_ASCENT  : ~0.8 Hz, higher amplitude vertical acceleration
      2 STAIR_DESCENT : ~0.8 Hz, brief impact spikes at heel-strike
      3 RAMP_ASCENT   : ~1 Hz, slight asymmetric accel (lean forward)
      4 RAMP_DESCENT  : ~1 Hz, slight forward lean, braking signals
      5 STANDING      : near-zero motion, small drift
      6 SIT_TO_STAND  : brief transient burst, then standing-like

    Channel layout (6-axis IMU):
      ch0: acc_x, ch1: acc_y, ch2: acc_z (m/s²)
      ch3: gyro_x, ch4: gyro_y, ch5: gyro_z (rad/s)

    Returns:
        X_list  : list of (n_samples, n_channels) float32 arrays
        y_list  : list of int class labels
        subjects: list of subject IDs (int, 1-based)
    """
    rng = np.random.default_rng(seed)
    n_samples = int(fs * duration_per_activity)
    t = np.linspace(0, duration_per_activity, n_samples, endpoint=False)

    # --- Per-activity signal profiles ---
    # Designed so that class-discriminative features survive bandpass filtering (0.5-40 Hz).
    # Key design rules:
    #   - Level walking: f0=1.8 Hz, moderate amplitude, symmetric
    #   - Stair ascent:  f0=0.75 Hz, high amp, strong h2/h3 harmonics, no impacts, gyro pitch up
    #   - Stair descent: f0=0.75 Hz, high amp, different harmonic ratio, heavy impacts (ch2 spikes)
    #   - Ramp ascent:   f0=1.1 Hz, moderate amp, strong gyro pitch (forward lean is AC-encoded)
    #   - Ramp descent:  f0=1.4 Hz, moderate amp, different freq from ramp ascent
    #   - Standing:      near zero after HP filter
    #   - Sit-to-stand:  brief burst only
    _profiles = {
        0: dict(freq=1.8,  acc_amp=2.0,  gyro_amp=0.8,  h2_ratio=0.3,  h3_ratio=0.15, impact=0.00, name='level_walking'),
        1: dict(freq=0.75, acc_amp=4.5,  gyro_amp=2.0,  h2_ratio=0.15, h3_ratio=0.08, impact=0.00, name='stair_ascent'),
        2: dict(freq=0.75, acc_amp=4.5,  gyro_amp=2.0,  h2_ratio=0.50, h3_ratio=0.30, impact=0.25, name='stair_descent'),
        3: dict(freq=1.1,  acc_amp=3.0,  gyro_amp=1.5,  h2_ratio=0.25, h3_ratio=0.10, impact=0.00, name='ramp_ascent'),
        4: dict(freq=1.4,  acc_amp=3.0,  gyro_amp=1.5,  h2_ratio=0.40, h3_ratio=0.20, impact=0.05, name='ramp_descent'),
        5: dict(freq=0.0,  acc_amp=0.10, gyro_amp=0.03, h2_ratio=0.0,  h3_ratio=0.0,  impact=0.00, name='standing'),
        6: dict(freq=0.0,  acc_amp=0.0,  gyro_amp=0.0,  h2_ratio=0.0,  h3_ratio=0.0,  impact=0.00, name='sit_to_stand'),
    }

    X_list, y_list, subjects = [], [], []

    for subj_id in range(1, n_subjects + 1):
        subj_seed = seed + subj_id * 1000
        rng_s = np.random.default_rng(subj_seed)

        # Subject-specific variation — only phase and tiny ±1% amp
        # (frequency and harmonic structure are preserved across subjects)
        subj_freq_scale = 1.0  # fixed — so spectral features transfer cross-subject
        subj_amp_scale  = 1.0 + rng_s.uniform(-0.02, 0.02)

        for cls_idx, prof in _profiles.items():
            sig = np.zeros((n_samples, n_channels), dtype=np.float32)

            if cls_idx == 6:
                # SIT_TO_STAND: brief 2-second burst at start, then settling
                burst_len = int(2.0 * fs)
                sig[:burst_len, 2] = (                   # vertical accel
                    rng_s.normal(0, 4.0, burst_len)
                    + 4.0 * np.sin(2 * np.pi * 0.5 * t[:burst_len])
                ).astype(np.float32)
                sig[:burst_len, 4] = rng_s.normal(0, 1.5, burst_len).astype(np.float32)
                # rest of signal is standing-like
                sig[burst_len:, :] = rng_s.normal(0, 0.15, (n_samples - burst_len, n_channels)).astype(np.float32)
                sig[burst_len:, 2] += 9.81  # gravity on z
            elif cls_idx == 5:
                # STANDING: small drift only
                for ch in range(n_channels):
                    drift = rng_s.normal(0, prof['acc_amp'] if ch < 3 else prof['gyro_amp'], n_samples)
                    sig[:, ch] = drift.astype(np.float32)
                sig[:, 2] += 9.81  # gravity
            else:
                f0 = prof['freq']
                # Subject-specific frequency offset — small ±3%
                f0_subj = f0 * subj_freq_scale

                h2 = prof['h2_ratio']
                h3 = prof['h3_ratio']

                # Accelerometer channels — forward(x), lateral(y), vertical(z)
                for ch in range(3):
                    amp = prof['acc_amp'] * subj_amp_scale * rng_s.uniform(0.95, 1.05)
                    phase = rng_s.uniform(0, 2 * np.pi)

                    sig[:, ch] = (
                        amp * np.sin(2 * np.pi * f0_subj * t + phase)
                        + amp * h2 * np.sin(2 * np.pi * 2 * f0_subj * t + phase)
                        + amp * h3 * np.sin(2 * np.pi * 3 * f0_subj * t + phase)
                        + rng_s.normal(0, 0.15, n_samples)
                    ).astype(np.float32)

                # Vertical (z): add gravity DC (not affected by bandpass but won't hurt RF)
                sig[:, 2] += 9.81

                # Impact spikes — large short-duration spikes at heel-strike (descents)
                if prof['impact'] > 0:
                    impact_prob_per_step = prof['impact']
                    # impacts occur once every stride (~1/f0 seconds)
                    impact_interval = int(fs / max(f0_subj, 0.5))
                    t_impact = impact_interval // 4  # start mid-first stride
                    while t_impact < n_samples:
                        if rng_s.uniform() < impact_prob_per_step * 1.5:
                            jitter = int(rng_s.uniform(-0.05, 0.05) * impact_interval)
                            ti = max(0, min(t_impact + jitter, n_samples - 1))
                            width = int(rng_s.uniform(0.008, 0.020) * fs)  # 8-20ms
                            end = min(ti + width, n_samples)
                            spike_amp = rng_s.uniform(5.0, 12.0)
                            # Impact is primarily in vertical acceleration (ch2)
                            sig[ti:end, 2] += float(spike_amp)
                            # Also appears in forward (ch0) during stair descent
                            sig[ti:end, 0] += float(spike_amp * 0.3)
                        t_impact += impact_interval

                # Gyroscope channels — correlated with step frequency
                for ch in range(3, 6):
                    amp = prof['gyro_amp'] * subj_amp_scale * rng_s.uniform(0.95, 1.05)
                    phase = rng_s.uniform(0, 2 * np.pi)
                    sig[:, ch] = (
                        amp * np.sin(2 * np.pi * f0_subj * t + phase)
                        + amp * h2 * np.sin(2 * np.pi * 2 * f0_subj * t + phase)
                        + rng_s.normal(0, 0.03, n_samples)
                    ).astype(np.float32)

            X_list.append(sig)
            y_list.append(cls_idx)
            subjects.append(subj_id)

    print(f"[synthetic] Generated {len(X_list)} segments × {n_samples} samples "
          f"× {n_channels} channels  ({n_subjects} subjects, {N_CLASSES} classes)")
    return X_list, y_list, subjects


# ── Real .mat data loading ──────────────────────────────────────────────────

def _get_subject_id(file_path: Path) -> int:
    """Extract subject ID (integer) from directory or filename."""
    for part in reversed(file_path.parts):
        lp = part.lower()
        if lp.startswith('subject') or lp.startswith('sub'):
            digits = ''.join(ch for ch in part if ch.isdigit())
            if digits:
                return int(digits)
    return 0


def _extract_signals(data: dict, n_channels: int = 6) -> np.ndarray | None:
    """
    Extract (n_samples, n_channels) float32 array from a loaded .mat dict.
    Tries common keys used in biomechanics datasets; falls back to first
    2-D array found.
    """
    try:
        import scipy.io  # noqa: F401 — already imported upstream
    except ImportError:
        return None

    preferred_keys = ['IMU', 'imu', 'Acc', 'acc', 'accel', 'Accel', 'data', 'signal']
    candidate = None

    # Try preferred keys first
    for key in preferred_keys:
        if key in data:
            val = data[key]
            if isinstance(val, np.ndarray) and val.ndim == 2:
                candidate = val
                break

    # Fall back to first 2-D numeric array
    if candidate is None:
        for key, val in data.items():
            if key.startswith('__'):
                continue
            if isinstance(val, np.ndarray) and val.ndim == 2 and val.size > 0:
                candidate = val
                break

    if candidate is None:
        return None

    # Ensure (n_samples, n_channels) orientation
    if candidate.shape[0] < candidate.shape[1]:
        candidate = candidate.T

    # Take up to n_channels columns
    candidate = candidate[:, :n_channels].astype(np.float32)
    return candidate


def load_real_data(data_dir: Path, n_channels: int = 6) -> tuple:
    """Load .mat files from data_dir and return (X_list, y_list, subjects)."""
    try:
        import scipy.io
    except ImportError:
        print("[data_loader] scipy not available — cannot load .mat files")
        return [], [], []

    mat_files = list(data_dir.rglob('*.mat'))
    if not mat_files:
        return [], [], []

    print(f"[data_loader] Found {len(mat_files)} .mat files under {data_dir}")
    X_list, y_list, subjects = [], [], []

    for f in sorted(mat_files):
        # Determine label from filename
        label = None
        fname_lower = f.stem.lower()
        for key, idx in ACTIVITY_MAP.items():
            if key in fname_lower:
                label = idx
                break
        if label is None:
            print(f"  Skipping (unknown activity): {f.name}")
            continue

        subj = _get_subject_id(f)

        try:
            data = scipy.io.loadmat(str(f))
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
            continue

        signals = _extract_signals(data, n_channels)
        if signals is None or signals.shape[0] < 10:
            print(f"  Skipping (no usable signal): {f.name}")
            continue

        X_list.append(signals)
        y_list.append(label)
        subjects.append(subj)
        print(f"  {f.name}: shape={signals.shape}, label={CLASS_NAMES[label]}, subj={subj}")

    return X_list, y_list, subjects


def load_dataset(
    data_dir: str = 'data/gait',
    n_channels: int = 6,
    fs: float = 1000.0,
) -> tuple:
    """
    Load HDsEMG+IMU dataset (real .mat files if present, else synthetic).

    Args:
        data_dir:   Path to directory containing .mat files.
        n_channels: Number of IMU channels to use (default 6).
        fs:         Sampling frequency in Hz (default 1000).

    Returns:
        X_list  : list of (n_samples, n_channels) float32 arrays, one per segment.
        y_list  : list of int class labels aligned with X_list.
        subjects: list of int subject IDs aligned with X_list.
        using_synthetic: bool — True if real data was not found.
    """
    data_path = Path(data_dir)
    X_list, y_list, subjects = load_real_data(data_path, n_channels)

    if len(X_list) == 0:
        print(f"[data_loader] No real data found in '{data_dir}'. "
              "Switching to SYNTHETIC data mode.")
        X_list, y_list, subjects = generate_synthetic_data(
            n_subjects=20,
            n_channels=n_channels,
            fs=fs,
            duration_per_activity=30.0,
        )
        return X_list, y_list, subjects, True

    # Verify all segments have enough channels
    valid = []
    for x, y, s in zip(X_list, y_list, subjects):
        if x.shape[1] < n_channels:
            # Pad with zeros if needed
            pad = np.zeros((x.shape[0], n_channels - x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        valid.append((x[:, :n_channels], y, s))
    X_list, y_list, subjects = zip(*valid) if valid else ([], [], [])
    X_list, y_list, subjects = list(X_list), list(y_list), list(subjects)

    print(f"[data_loader] Loaded {len(X_list)} real segments from {data_dir}")
    return X_list, y_list, subjects, False


# ── Dataset explorer ────────────────────────────────────────────────────────

def explore_dataset(data_dir: str = 'data/gait') -> None:
    """Print structure of first found .mat file."""
    try:
        import scipy.io
    except ImportError:
        print("scipy not installed. Cannot load .mat files.")
        return

    data_path = Path(data_dir)
    mat_files = list(data_path.rglob('*.mat'))
    if not mat_files:
        print(f"No .mat files found in '{data_dir}'.")
        print("Run with synthetic data instead:")
        print("  X_list, y_list, subjects, _ = load_dataset()")
        return

    print(f"\n=== Dataset Explorer ({data_dir}) ===")
    print(f"Total .mat files: {len(mat_files)}")
    print(f"\nSample: {mat_files[0]}")

    sample = scipy.io.loadmat(str(mat_files[0]))
    print("Keys and shapes:")
    for k, v in sample.items():
        if k.startswith('__'):
            continue
        if hasattr(v, 'shape'):
            print(f"  '{k}': shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  '{k}': {type(v).__name__} = {v!r}")

    # Print class distribution
    label_counts = {}
    for f in mat_files:
        for key, idx in ACTIVITY_MAP.items():
            if key in f.stem.lower():
                label_counts[CLASS_NAMES[idx]] = label_counts.get(CLASS_NAMES[idx], 0) + 1
                break

    if label_counts:
        print("\nActivity distribution (by file count):")
        for name, cnt in sorted(label_counts.items()):
            print(f"  {name:20s}: {cnt} files")


if __name__ == '__main__':
    # Quick smoke-test
    explore_dataset()

    print("\n--- Synthetic data test ---")
    X, y, subj, synthetic = load_dataset(data_dir='data/gait')
    print(f"using_synthetic={synthetic}")
    print(f"Segments: {len(X)}, labels: {set(y)}")
    print(f"First segment shape: {X[0].shape}")
    print(f"Subjects: {sorted(set(subj))}")

    # Class counts
    from collections import Counter
    counts = Counter(y)
    for cls_idx in sorted(counts):
        print(f"  {CLASS_NAMES[cls_idx]:20s} (cls {cls_idx}): {counts[cls_idx]} segments")
