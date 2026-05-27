"""
classifier.py — Model definitions for gait intent recognition.

Models:
  1. Random Forest (sklearn) — fast baseline on hand-crafted features
  2. GaitCNN1D (PyTorch)    — 1-D convolutional network on raw windows
"""

from sklearn.ensemble import RandomForestClassifier

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[classifier] PyTorch not installed — CNN model unavailable. RF only.")


# ── Random Forest ────────────────────────────────────────────────────────────

def build_random_forest() -> RandomForestClassifier:
    """
    Build the RF baseline classifier.

    Hyper-parameters chosen for gait classification:
      - 200 trees with max_depth=20 → enough capacity for 7 classes
      - class_weight='balanced' → handles class imbalance in synthetic data
      - n_jobs=-1 → use all CPU cores
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
    )


# ── 1D CNN ───────────────────────────────────────────────────────────────────

if TORCH_OK:
    class GaitCNN1D(nn.Module):
        """
        1-D Convolutional Neural Network for locomotion mode classification.

        Architecture:
            Input : (batch, window_size, n_channels)   [permuted internally]
            Block1: Conv1d(n_ch→32, k=5, pad=2) → BN → ReLU → MaxPool(2)
            Block2: Conv1d(32→64,   k=3, pad=1) → BN → ReLU → MaxPool(2)
            Block3: Conv1d(64→128,  k=3, pad=1) → BN → ReLU
            GAP   : AdaptiveAvgPool1d(1) → flatten → (batch, 128)
            Head  : Linear(128) → ReLU → Dropout(0.3) → Linear(n_classes)

        Also uses magnitude spectrum of the input as an auxiliary feature to
        help with frequency-domain discrimination (phase-invariant).

        Output: raw logits (batch, n_classes).  Apply softmax for probabilities.
        """

        def __init__(
            self,
            n_channels: int,
            n_classes: int = 7,
            window_size: int = 200,
        ):
            super().__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.window_size = window_size
            self.fft_bins = window_size // 2 + 1  # rfft output size

            # Time-domain branch
            self.time_features = nn.Sequential(
                # Block 1
                nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),          # T/2

                # Block 2
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),          # T/4

                # Block 3
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
            )

            # Frequency-domain branch (magnitude spectrum — phase invariant)
            self.freq_features = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
            )

            # Global average pool
            self.global_pool      = nn.AdaptiveAvgPool1d(1)
            self.global_pool_freq = nn.AdaptiveAvgPool1d(1)

            # Combined head: 128 (time) + 64 (freq) → n_classes
            self.classifier = nn.Sequential(
                nn.Linear(128 + 64, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            # x: (batch, window_size, n_channels)
            xp = x.permute(0, 2, 1)                     # → (batch, n_ch, T)

            # Time-domain path
            t_feat = self.time_features(xp)              # → (batch, 128, T/4)
            t_feat = self.global_pool(t_feat).squeeze(-1)  # → (batch, 128)

            # Frequency-domain path: compute magnitude spectrum
            # rfft returns complex tensor; take absolute value
            xf = torch.fft.rfft(xp, dim=-1)             # → (batch, n_ch, T//2+1)
            xf_mag = xf.abs() / (self.window_size ** 0.5)  # normalize
            f_feat = self.freq_features(xf_mag)          # → (batch, 64, ...)
            f_feat = self.global_pool_freq(f_feat).squeeze(-1)  # → (batch, 64)

            combined = torch.cat([t_feat, f_feat], dim=1)  # → (batch, 192)
            return self.classifier(combined)              # → (batch, n_classes)

        # Alias for backward compat with original architecture
        @property
        def features(self):
            return self.time_features

        def predict_proba(self, x: 'torch.Tensor') -> 'torch.Tensor':
            """Return softmax probabilities."""
            return torch.softmax(self.forward(x), dim=-1)

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:
    # Stub class so imports don't break when torch is missing
    class GaitCNN1D:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for GaitCNN1D. "
                "Install with: pip install torch"
            )


# ── Convenience factory ───────────────────────────────────────────────────────

def build_cnn(n_channels: int, n_classes: int = 7, window_size: int = 200):
    """Build a GaitCNN1D and move to best available device."""
    if not TORCH_OK:
        raise ImportError("PyTorch not available.")
    model = GaitCNN1D(n_channels=n_channels, n_classes=n_classes, window_size=window_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device), device


if __name__ == '__main__':
    print("=== classifier.py self-test ===")
    rf = build_random_forest()
    print(f"RF: {rf}")

    if TORCH_OK:
        import torch
        model, device = build_cnn(n_channels=6)
        print(f"CNN device: {device}")
        print(f"CNN parameters: {model.count_parameters():,}")
        dummy = torch.randn(4, 200, 6).to(device)
        out = model(dummy)
        print(f"CNN output shape: {out.shape}")  # (4, 7)
    else:
        print("PyTorch not available — CNN test skipped.")
