# Guide 02 — Medical Datasets

## Selected Datasets

### Option 1: Mus-V (Multimodal Ultrasound Vascular Segmentation)
### Option 2: HDsEMG+IMU (High-density EMG + IMU + Kinetics)

---

## Option 1: Mus-V Dataset

**Full name:** Multimodal Ultrasound Vascular Segmentation Dataset  
**Source paper:** "Force Sensing Guided Artery-Vein Segmentation via Sequential Ultrasound Images" — MICCAI 2024  
**Kaggle URL:** https://www.kaggle.com/datasets/among22/multimodal-ultrasound-vascular-segmentation  
**Score:** 10/10

### Content
| Property | Value |
|----------|-------|
| Modality | B-mode ultrasound (2D grayscale) |
| Anatomy | Carotid artery, femoral artery, jugular vein, femoral vein |
| Images | 3,114 ultrasound frames from 105 video sequences |
| Subjects | 10 volunteers |
| Annotations | Binary segmentation masks (artery class, vein class separately) |
| Extra | Force sensor readings from ultrasound probe (multimodal) |
| Size | ~200 MB |
| License | Academic use (Kaggle terms) |

### Why This Dataset
- **Femoral artery** is explicitly included — directly matches the assignment target
- Artery and vein are **separately labeled** — critical for the robot to target only the artery
- **Real ultrasound images** with authentic speckle noise — realistic for clinical scenario
- Sufficient data (3,114 images) for demonstrating detection pipeline
- Freely available without institutional approval

### Download

```bash
# Install kaggle CLI
pip install kaggle

# Configure API key (get from https://www.kaggle.com/settings/account)
mkdir -p ~/.kaggle
# Put your kaggle.json here: ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download among22/multimodal-ultrasound-vascular-segmentation \
  -p /home/gokul/github_repos/cyient/data/sample_images/ \
  --unzip

# Verify
ls /home/gokul/github_repos/cyient/data/sample_images/
```

Alternatively, download manually from:
https://www.kaggle.com/datasets/among22/multimodal-ultrasound-vascular-segmentation

### Expected File Structure After Download
```
data/sample_images/
├── images/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── masks/
│   ├── artery/
│   │   ├── 0001.png
│   │   └── ...
│   └── vein/
│       ├── 0001.png
│       └── ...
└── metadata.csv  (if provided)
```

> **Note:** If the exact structure differs from above, adapt `image_publisher.py` to match.
> Use `os.listdir()` to explore the actual structure first.

### How We Use It
- Load 3–5 sample images for the demo (no training needed — classical CV approach)
- Use artery masks as ground truth to verify detection
- Image dimensions will be read dynamically by the perception node

---

## Option 2: HDsEMG+IMU Dataset

**Full name:** High-density EMG, IMU, kinetic, and kinematic open-source data for comprehensive locomotion activities  
**Source paper:** Nature Scientific Data (2023)  
**DOI:** https://doi.org/10.6084/m9.figshare.22227337  
**Paper:** https://www.nature.com/articles/s41597-023-02679-x  
**Score:** 10/10

### Content
| Property | Value |
|----------|-------|
| Sensor types | 64-channel HD-sEMG (dominant leg), 6-axis IMU, force plates, motion capture |
| Locomotion modes | Level walking (3 speeds), slope ascent/descent (3 speeds each), stair ascent/descent (3 speeds each), side-stepping, transitions |
| Total activities | 8+ distinct locomotion categories |
| Subjects | 10 healthy adults |
| Sampling rate | 2000 Hz (EMG), 1000 Hz (IMU), 1000 Hz (kinetics) |
| Duration | Multi-hour per subject across all conditions |
| Size | ~2–5 GB (multiple files) |
| License | CC0 (public domain) |
| Format | `.mat` files (MATLAB) / possible CSV exports |

### Why This Dataset
- Includes **all locomotion modes needed for an exoskeleton FSM**: walking (3 speeds), stair ascent/descent, ramp ascent/descent, sit-to-stand, transitions
- **Both EMG and IMU** — EMG provides ~100–200 ms early intent signal before motion, IMU confirms motion state
- **Speed variations** per activity — critical for realistic control (slow vs. fast walking differs biomechanically)
- CC0 license — no access barriers
- **Nature Scientific Data** publication — trusted, well-documented

### Download

```bash
# Method 1: Figshare direct download (recommended)
pip install requests  # if not installed

python3 - << 'EOF'
import requests, os, zipfile

# Figshare article ID
article_id = "22227337"
base_url = f"https://api.figshare.com/v2/articles/{article_id}/files"

resp = requests.get(base_url)
files = resp.json()
os.makedirs("data/gait", exist_ok=True)

for f in files:
    print(f"Downloading: {f['name']} ({f['size']//1024//1024} MB)")
    r = requests.get(f['download_url'], stream=True)
    with open(f"data/gait/{f['name']}", 'wb') as fp:
        for chunk in r.iter_content(chunk_size=8192):
            fp.write(chunk)
print("Done.")
EOF

# Method 2: Manual download
# Visit: https://figshare.com/articles/dataset/22227337
# Click each file to download
# Save to: data/gait/
```

### Expected File Structure
```
data/gait/
├── Subject01/
│   ├── level_walking_slow.mat
│   ├── level_walking_normal.mat
│   ├── level_walking_fast.mat
│   ├── stair_ascent_slow.mat
│   ├── stair_ascent_normal.mat
│   ├── stair_descent_slow.mat
│   ├── ramp_ascent_slow.mat
│   ├── ramp_descent_slow.mat
│   └── ...
├── Subject02/
│   └── ...
└── README.md (dataset documentation)
```

> **Note:** Exact structure depends on the Figshare upload. Use `scipy.io.loadmat()` to load `.mat` files.
> Run `python3 -c "import scipy.io; d=scipy.io.loadmat('data/gait/...'); print(d.keys())"` to explore.

### Loading `.mat` Files in Python

```python
import scipy.io
import numpy as np

data = scipy.io.loadmat('data/gait/Subject01/level_walking_slow.mat')

# Common keys in biomechanics datasets:
# 'EMG'    -> (n_samples, n_channels) array
# 'IMU'    -> (n_samples, n_imu_channels)
# 'labels' -> (n_samples,) activity labels
# 'fs'     -> sampling frequency scalar

for key in data.keys():
    if not key.startswith('__'):
        val = data[key]
        if hasattr(val, 'shape'):
            print(f"{key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"{key}: {val}")
```

### Activity-to-Label Mapping
```python
ACTIVITY_MAP = {
    'level_walking':    0,   # LEVEL_WALKING
    'stair_ascent':     1,   # STAIR_ASCENT
    'stair_descent':    2,   # STAIR_DESCENT
    'ramp_ascent':      3,   # RAMP_ASCENT
    'ramp_descent':     4,   # RAMP_DESCENT
    'standing':         5,   # STANDING
    'sit_to_stand':     6,   # SIT_TO_STAND
}
CLASS_NAMES = list(ACTIVITY_MAP.keys())
```

---

## Alternative Datasets (Reference Only — Not Used)

### Option 1 Alternatives

| Dataset | Score | Why Not Selected |
|---------|-------|-----------------|
| ROSE (OCTA Retinal) | 7/10 | Wrong modality (OCTA) and anatomy (retina) |
| fUS Rat Cerebral | 6/10 | Animal model, access-restricted |
| DRIVE (Retinal Fundus) | 6/10 | Fundus photography ≠ ultrasound; retina ≠ femoral |

### Option 2 Alternatives

| Dataset | Score | Why Not Selected |
|---------|-------|-----------------|
| Gait120 (Nature 2025) | 9/10 | Very new (2025), access unclear |
| ENABL3S | 9/10 | Slightly smaller scope; HDsEMG includes everything it has plus more |
| UCI HAR | 7/10 | Only 6 activities, no speed variation, no transitions |

---

## Dataset Scoring Methodology

Datasets were scored on 5 criteria (0–2 points each, max 10):

| Criterion | 2 pts | 1 pt | 0 pts |
|-----------|-------|-------|-------|
| **Task Fit** | Direct match (e.g., femoral US / exo locomotion) | Related but different anatomy/modality | Unrelated |
| **Ease of Access** | Free, direct download, no registration | Free with registration/contact | Paid or approval required |
| **Size/Richness** | Ideal range (100–10k samples or 5–30 subjects) | Too small or too large | Unusable |
| **Label Quality** | Pixel-perfect masks or precise activity timestamps | Coarse/partial labels | No labels |
| **Documentation** | Peer-reviewed paper + format docs | Basic README | Undocumented |

This scoring system prioritizes **usability within a 2-day sprint** while ensuring the data is appropriate for the clinical application.
