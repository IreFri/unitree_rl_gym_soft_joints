# H1 Humanoid Config & Five Experimental Setups

This README explains how to toggle the **five configurations** used in our experiments by editing the flags at the top of `h1_config.py`. These flags control whether stiffness/damping are **fixed** or **variable**, which **preset** is used for fixed stiffness, and whether stiffness is applied in **parallel** or **series** with the joint.

> File to edit: `h1_config.py` (top of the file)

```python
# Define control model flags (choose one set at a time)
VARIABLE_STIFFNESS = True                   # If True → variable stiffness/damping are actions from the policy
FIXED_STIFFNESS   = not VARIABLE_STIFFNESS  # Automatically inferred

# If FIXED_STIFFNESS → choose soft or rigid preset
FIXED_STIFFNESS_PRESET = 'rigid'            # Options: 'soft' or 'rigid'

# Choose stiffness model
PARALLEL = True   # stiffness in parallel configuration
SERIE    = False  # stiffness in series configuration
```

---

## The Five Configurations (What to Set)

| Case | Description                                                   | `VARIABLE_STIFFNESS` | `FIXED_STIFFNESS_PRESET` | `PARALLEL` | `SERIE` |
|:----:|---------------------------------------------------------------|:---------------------:|:-------------------------:|:----------:|:------:|
| a)   | **Rigid joints** (baseline torque control; k=0, d=0)         | `False`               | `'rigid'`                 | `True`     | `False` |
| b)   | **Soft, fixed (parallel)**                                   | `False`               | `'soft'`                  | `True`     | `False` |
| c)   | **Soft, variable (parallel)** — RL learns k,d                 | `True`                | _ignored_                 | `True`     | `False` |
| d)   | **Soft, fixed (series)**                                     | `False`               | `'soft'`                  | `False`    | `True`  |
| e)   | **Soft, variable (series)** — RL learns k,d                   | `True`                | _ignored_                 | `False`    | `True`  |

**Notes**  
- `FIXED_STIFFNESS` is automatically `not VARIABLE_STIFFNESS`. You do **not** need to set it manually.  
- For **fixed** setups (b, d, a), `FIXED_STIFFNESS_PRESET` selects either `'soft'` or `'rigid'` gains.  
- Exactly **one** of `PARALLEL` / `SERIE` must be `True`. Keep the other `False`.  
- For **rigid** (a), we still select `PARALLEL=True` for consistency, but `k=d=0` so no extra torque is added.  
- For **variable** setups (c, e), the policy action includes 10 stiffness + 10 damping values (one per leg DOF), bounded by `max_stiffness` and `max_damping` in the config.

---

## Where These Flags Take Effect

- `class control` in `h1_config.py` uses:
  - `variable_stiffness` vs `fixed_stiffness`
  - `fixed_stiffness_values` when fixed (`'soft'` or `'rigid'` preset)
  - `stiffness_model ∈ {"parallel","series"}` and `fixed_stiffness_model` accordingly
  - `max_stiffness`, `max_damping` for variable setups

- `h1_env.py` then applies the model:
  - **Parallel**: adds `k(q_ref - q)` and `-d q_dot` to torque control (in addition to base torque)
  - **Series**: modulates a virtual motor position `q_m` and computes `k(q_m - q) - d q_dot`

---

## Quick Switch Examples

> Edit the flags in `h1_config.py`, then run your script (e.g., `play.py`, `play_for_timedata.py`, `evaluate_on_terrains.py`, `evaluate_online_video_attractors.py`).

### a) Rigid Joints (baseline)
```python
VARIABLE_STIFFNESS = False
FIXED_STIFFNESS_PRESET = 'rigid'
PARALLEL = True
SERIE = False
```

### b) Soft, Fixed (Parallel)
```python
VARIABLE_STIFFNESS = False
FIXED_STIFFNESS_PRESET = 'soft'
PARALLEL = True
SERIE = False
```

### c) Soft, Variable (Parallel)
```python
VARIABLE_STIFFNESS = True
PARALLEL = True
SERIE = False
# FIXED_STIFFNESS_PRESET ignored
```

### d) Soft, Fixed (Series)
```python
VARIABLE_STIFFNESS = False
FIXED_STIFFNESS_PRESET = 'soft'
PARALLEL = False
SERIE = True
```

### e) Soft, Variable (Series)
```python
VARIABLE_STIFFNESS = True
PARALLEL = False
SERIE = True
# FIXED_STIFFNESS_PRESET ignored
```

---

## Typical Run Commands


```bash
# Play a trained policy (headless)
python play.py --robot=H1_soft --load_run=H1_soft_serie --checkpoint=15000 --stiffness_model=series --headless

# Evaluate and record video (attractors)
python evaluate_online_video_attractors.py --robot=H1_soft --load_run=H1_soft_serie --checkpoint=15000 --stiffness_model=series

# Evaluate on terrains and export CSV metrics
python evaluate_on_terrains.py --robot=H1_normal --checkpoint=10000 --terrains=plane,stones --out_dir=plots/metrics
```

> Make sure your **URDF** and **checkpoints** are available and that `legged_gym` and Isaac Gym are properly installed.

---

