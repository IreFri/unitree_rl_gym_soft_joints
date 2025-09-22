# Humanoid Walking with Variable Stiffness
This repository contains scripts used for evaluating and visualizing humanoid robot walking experiments with rigid joints, soft joints, and variable stiffness/damping. The code is built on top of [Isaac Gym](https://developer.nvidia.com/isaac-gym) and the `unitree_rl_gym` / `legged_gym` framework.

## 📦 Installation

1. **Dependencies**  
   - Python 3.8+  
   - [Isaac Gym (preview 4)](https://developer.nvidia.com/isaac-gym) installed and working  
   - PyTorch with CUDA (for GPU)  
   - Other packages (install with pip):  
     ```bash
     pip install -r requirements.txt
     ```
     Typical requirements:
     ```
     numpy
     pandas
     matplotlib
     opencv-python
     PyQt5   # optional, for GUI
     ```
   
2. **Environment**  
   We recommend running inside a conda environment (example name: `isaac11`):  
   ```bash
   conda create -n isaac11 python=3.8
   conda activate isaac11
   ```

---

## Scripts Overview

All scripts are located in `scripts/`. You can run them with:

```bash
python <script_name>.py [options]
```

### 1. `train.py`
Train a new RL policy for the humanoid robot. Saves checkpoints under `logs/`.

**Example:**
```bash
python train.py --robot=H1_normal --run_name=H1_normal_run
```

---

### 2. `play.py`
Run a trained policy in Isaac Gym (with or without GUI), optionally recording video.

**Example:**
```bash
python play.py --robot=H1_soft --load_run=H1_soft_serie --checkpoint=15000 --stiffness_model=series --headless
```

---

### 3. `play_for_timedata.py`
Run a trained policy and log **time-series data** (CoM, forces, joint states, etc.) to CSV for later analysis.

**Example:**
```bash
python play_for_timedata.py --robot=H1_var --load_run=H1_var_parallel --checkpoint=20000 --stiffness_model=parallel --out_dir=plots/timedata
```

---

### 4. `play_push.py`
Run a trained policy and apply an **external push** to test recovery. Push timing, magnitude, and direction are configurable.

**Example:**
```bash
python play_push.py --robot=H1_soft --checkpoint=12000 --force=200 --t0=5.0 --duration=0.2
```

---

### 5. `evaluate_on_terrains.py`
Run evaluations across **different terrain types** (plane, stepping stones, uneven ground, etc.). Logs metrics such as success rate, energy, and stability to CSV.

**Example:**
```bash
python evaluate_on_terrains.py --robot=H1_normal --checkpoint=10000 --terrains=plane,stones
```

---

### 6. `evaluate_online_video_attractors.py`
Evaluate policies and record videos of attractor-based walking (different stiffness models). Outputs CSV logs and MP4 videos.

**Example:**
```bash
python evaluate_online_video_attractors.py --robot=H1_soft --load_run=H1_soft_serie --checkpoint=15000 --stiffness_model=series
```

---

##  Outputs

- **Plots, CSVs, and videos** are stored under:
  ```
  plots/
    ├── data/...
    ├── videos/...
    └── figures/...
  ```
- Each evaluation script creates its own subdirectory with results.

---

##  Notes

- By default, scripts assume **headless mode** for reproducibility. Use `--gui` or `--record_frames` for visualizations.
- Large datasets and models are **not** included in this repo. Only lightweight configs and sample data are provided.
- The code depends on `legged_gym` and `unitree_rl_gym`. Ensure they are installed and accessible via `PYTHONPATH`.

---

##  Citation
If you use this code in academic work, please cite:

```
Title: "Compliant Leg Joints for Humanoids: Parallel vs. Series Elasticity in RL-Based Locomotion"
Authors: Irene Frizza, Max Austin, Kohei Nakajima. 
```
