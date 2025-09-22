import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import glob
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import torch_utils 
import isaacgym.torch_utils as torch_utils
import isaacgym.gymtorch as gymtorch
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import subprocess
import json
import time
import cv2 
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QSlider, QLabel
from PyQt5.QtCore import Qt
sys.path.append("legged_gym/utils")
from terrain import Terrain  
import legged_gym.utils.terrain as terrain_module
from terrain import Terrain as CustomTerrain
import pandas as pd
import argparse
import math
from sklearn.decomposition import PCA
from matplotlib import cm
import matplotlib.pyplot as plt 



EXPORT_POLICY = False
RECORD_FRAMES = False
MOVE_CAMERA = False
BREAK_WHEN_FALL = False

CAMERA_VIEW = "BACK"  # Options: "BACK", "FRONT", "RIGHT", "LEFT", "TOP"

plots_dir = "legged_gym/plots"
video_out_path_hq = os.path.join(plots_dir, "simulation_video_HQ.mp4")
json_path = os.path.join(plots_dir, "robot_push_var_750.json")

# Storage for analysis
com_history = []
zmp_history = []
support_polygon_history = []
motor_torque_history = []
sensor_torque_history = []
feet_contact_history = []
joint_angle_history = []
frame_times = []
current_time = time.time()
   
def convert_tensors(obj):
    """ Recursively convert torch tensors, numpy arrays, and numpy scalars to Python-native types. """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors(i) for i in obj]
    return obj


def _init_csv(path, columns):
    """Create/overwrite a CSV with header."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)

def _append_csv(path, df):
    """Append to CSV without header if file already exists."""
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    df.to_csv(path, mode='a', index=False, header=write_header)


def evaluate_policy(args, load_run, urdf_override=None, terrains=None, custom_terrain_kwargs=None):
    
    args.headless = False
    args.task = "h1"
    all_results = []

    global LOAD_RUN
    LOAD_RUN = load_run

    if urdf_override:
        os.environ["URDF_OVERRIDE_PATH"] = urdf_override

    for terrain_type in terrains:
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        env_cfg.domain_rand.push_robots = False

        if args.robot == "H1_var":
            env_cfg.control.variable_stiffness = True
            env_cfg.control.fixed_stiffness = False
            env_cfg.control.stiffness_model = STIFFNESS_MODEL
        elif args.robot == "H1_soft":
            env_cfg.control.variable_stiffness = False
            env_cfg.control.fixed_stiffness = True
            env_cfg.control.stiffness_model = STIFFNESS_MODEL
            env_cfg.control.fixed_stiffness_values = {
                'stiffness': [200.0, 100.0, 80.0, 200.0, 100.0] * 2,
                'damping':   [4.5, 4.5, 1.0, 5.0, 4.0] * 2
            }
        elif args.robot == "H1_normal":
            env_cfg.control.variable_stiffness = False
            env_cfg.control.fixed_stiffness = True
            env_cfg.control.stiffness_model = STIFFNESS_MODEL
            env_cfg.control.fixed_stiffness_values = {
                'stiffness': [0.0] * 10,
                'damping':   [0.0] * 10
            }
        else:
            raise ValueError("Unknown robot type. Use H1_normal, H1_soft, or H1_var.")

        train_cfg.runner.load_run = load_run
        train_cfg.runner.checkpoint = CHECKPOINT
        
        # 👉 Pass custom_terrain_kwargs to play
        results = play(args, terrain_type, custom_terrain_kwargs=custom_terrain_kwargs)

        for task_name, task_metrics in results.items():
            num_trials = len(task_metrics["success"])
            for trial in range(num_trials):
                row = {
                    "robot": load_run,
                    "terrain": terrain_type.split(".")[-1],
                    "task": task_name,
                }
                for key, values in task_metrics.items():
                    # ➡️ First, check if the current key has enough entries for this trial
                    if not isinstance(values, list) or trial >= len(values):
                        print(f"⚠️ Warning: Missing value for key '{key}' at trial {trial}. Skipping.")
                        continue

                    if key in ["com_full", "zmp_full"]:
                        row[key] = json.dumps(convert_tensors(values[trial]))  # Save as JSON string
                    elif isinstance(values[trial], list):
                        row.update({f"{key}_{i}": v for i, v in enumerate(values[trial])})
                    else:
                        row[key] = values[trial]

                all_results.append(row)

    df = pd.DataFrame(all_results)
    return df



def warmup(env, actor_critic, num_steps=20):
    """
    Warmup the robot: reset pose, stop velocities, stabilize before starting real motion.
    """
    # 0. Reset robot velocities
    gym_root_states_raw = env.gym.acquire_actor_root_state_tensor(env.sim)
    gym_root_states = gymtorch.wrap_tensor(gym_root_states_raw).view(-1, 13)

    # Set linear and angular velocities to 0
    gym_root_states[:, 7:13] = 0.0

    env.gym.set_actor_root_state_tensor(env.sim, gym_root_states_raw)

    # 1. Reset DOF positions and velocities
    gym_dof_states_raw = env.gym.acquire_dof_state_tensor(env.sim)
    gym_dof_states = gymtorch.wrap_tensor(gym_dof_states_raw).view(env.num_envs * env.num_dof, 2)
    
    gym_dof_states[:, 0] = env.default_dof_pos.view(-1)  # Default pose
    gym_dof_states[:, 1] = 0.0                           # Zero velocities

    env.gym.set_dof_state_tensor(env.sim, gym_dof_states_raw)

    # 2. Set zero command to stop robot
    env.commands[:] = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=env.device)

    # 3. Take some stabilization steps
    obs, _ = env.get_observations()
    for _ in range(num_steps):
        with torch.no_grad():
            actions = actor_critic.act_inference(obs)  # Use policy (standing policy)
        obs, _, _, _, _ = env.step(actions.detach())

    print(f"🔥 Warmup finished: {num_steps} stabilization steps (robot should now be standing)")





def correct_lateral_velocity(env):
    import isaacgym.gymtorch as gymtorch

    # Acquire the root states
    gym_root_states_raw = env.gym.acquire_actor_root_state_tensor(env.sim)
    gym_root_states = gymtorch.wrap_tensor(gym_root_states_raw).view(-1, 13)

    # Get linear and angular velocities
    lin_vel = env.root_states[:, 7:10]
    ang_vel = env.root_states[:, 10:13]

    # print(f"Before correction: vx={lin_vel[0,0].item():.3f} vy={lin_vel[0,1].item():.3f} vz={lin_vel[0,2].item():.3f} | yaw_rate={ang_vel[0,2].item():.3f}")

    lin_vel[:, 0] *= 0.0  # Zero forward drift
    lin_vel[:, 1] *= 0.0  # Zero lateral drift
    ang_vel[:, 2] *= 0.0  # Zero yaw drift

    # Apply corrections
    env.root_states[:, 7:10] = lin_vel
    env.root_states[:, 10:13] = ang_vel

    # Push back corrected states
    gym_root_states.copy_(env.root_states)
    env.gym.set_actor_root_state_tensor(env.sim, gym_root_states_raw)

    # print(f"✅ After correction: vx={lin_vel[0,0].item():.3f} vy={lin_vel[0,1].item():.3f} vz={lin_vel[0,2].item():.3f} | yaw_rate={ang_vel[0,2].item():.3f}")


def adjust_yaw_command(desired_heading, env):

    if desired_heading is None:
        return 0.0 
    current_yaw = env.rpy[0][2].item()
    heading_error = desired_heading - current_yaw
    corrected_yaw_rate = heading_error * 2.0

    return corrected_yaw_rate



def _update_camera(env, top_view_offset=0.0):
    """ Move the camera smoothly during push recovery. """
    if env.viewer is not None:
        com_pos = env.root_states[:, 0:3].detach().cpu().numpy()
        robot_pos = com_pos[0].tolist()

        # 👇 Raise camera height by top_view_offset
        camera_position = gymapi.Vec3(
            robot_pos[0] - 2.0,
            robot_pos[1],
            robot_pos[2] + 1.0 + top_view_offset  # 👈 shift camera higher
        )
        camera_target = gymapi.Vec3(robot_pos[0], robot_pos[1], robot_pos[2])

        env.gym.viewer_camera_look_at(env.viewer, None, camera_position, camera_target)

def _record_rnn_state(actor_critic, hidden_hist, cell_hist):
    """
    Appends one timestep of hidden (and possibly cell) states for env 0.
    Shapes expected from rsl_rl ActorCriticRecurrent:
      - LSTM: (num_layers, num_envs, hidden_dim) for both h and c
      - GRU:  (num_layers, num_envs, hidden_dim) for h
    """
    hidden_states, _ = actor_critic.get_hidden_states()
    if isinstance(hidden_states, tuple):  # LSTM
        hx, cx = hidden_states
        # capture ALL layers for env 0 at this timestep
        hx_list = hx[:, 0, :].detach().cpu().numpy().tolist()
        cx_list = cx[:, 0, :].detach().cpu().numpy().tolist()
        hidden_hist.append(hx_list)
        cell_hist.append(cx_list)
    else:  # GRU or simple RNN
        hx = hidden_states
        if hx.dim() == 3:
            hx_list = hx[:, 0, :].detach().cpu().numpy().tolist()
        else:
            # fallback if shape is [num_envs, hidden_dim]
            hx_list = hx[0].detach().cpu().numpy().tolist()
        hidden_hist.append(hx_list)
        # keep cell history aligned; use None for GRU
        cell_hist.append(None)


class PhaseLockedTrigger:
    """
    Phase-locked push trigger.
    - support_foot: "left" or "right" (stance foot whose phase you align to)
    - target: phase in [0,1) at which to trigger (e.g., 0.50 = mid-stance)
    - tol: ±window around the target to allow triggering
    - warmup_steps: minimum steps before we start checking
    - contact_N: contact force threshold (N) to declare foot contact
    - hold_frames: require contact pattern to be stable for a few frames
    """
    def __init__(self, support_foot="left", target=0.50, tol=0.02,
                 warmup_steps=200, contact_N=30.0, hold_frames=3):
        self.support_foot = support_foot.lower()
        self.target = float(target)
        self.tol = float(tol)
        self.warmup_steps = int(warmup_steps)
        self.contact_N = float(contact_N)
        self.hold_frames = int(hold_frames)

        self.prev_L = False
        self.prev_R = False
        self.last_hs_t = None
        self.step_period = None
        self.phase = 0.0
        self.stable_ctr = 0
        self.t = 0

    def _heel_strike(self, prev_c, cur_c):
        return (not prev_c) and cur_c

    def __call__(self, env):
        """Update phase state from env and return True when target phase window is hit."""
        # --- read contacts ---
        try:
            left_z  = env.contact_forces[0, env.left_foot_indices,  2].max().item()
            right_z = env.contact_forces[0, env.right_foot_indices, 2].max().item()
        except Exception:
            left_z, right_z = 0.0, 0.0

        L = (left_z  > self.contact_N)
        R = (right_z > self.contact_N)

        # which foot defines stance/phase?
        if self.support_foot == "left":
            hs = self._heel_strike(self.prev_L, L)
            in_stance = L
        else:
            hs = self._heel_strike(self.prev_R, R)
            in_stance = R

        # time index (in steps)
        self.t += 1

        # estimate step period from heel-strike intervals (in steps)
        if hs:
            if self.last_hs_t is not None:
                self.step_period = max(1, self.t - self.last_hs_t)
            self.last_hs_t = self.t
            self.phase = 0.0
            self.stable_ctr = 0
        else:
            if self.step_period is not None and in_stance:
                # advance normalized phase during stance
                self.phase = min(0.999, self.phase + 1.0 / self.step_period)

        self.prev_L, self.prev_R = L, R

        # require some warmup and a small stability window to avoid single-frame glitches
        if self.t < self.warmup_steps or self.step_period is None or not in_stance:
            self.stable_ctr = 0
            return False

        # inside phase window?
        if abs(self.phase - self.target) <= self.tol:
            self.stable_ctr += 1
            if self.stable_ctr >= self.hold_frames:
                return True
        else:
            self.stable_ctr = 0
        return False



from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm

class OnlineLSTMAttractor3D:
    """
    Live 3D PCA of the LSTM hidden state with a time-colored trajectory,
    horizontal colorbar (Start | Push | End), and optional 'push' marker.
    API mirrors OnlineLSTMPlot.update(...) so you can call it the same way.
    """
    def __init__(self, title="LSTM attractor (PCA, live)", expect_steps=1000, decim_draw=2):
        self.hidden_hist = []          # raw h_t history (list of 1D arrays)
        self.push_idx = None
        self.decim_draw = max(1, int(decim_draw))
        self._step_idx = 0

        # Matplotlib setup
        plt.ion()
        self.fig = plt.figure(figsize=(5, 4), constrained_layout=True)
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("PC1", fontsize=12)
        self.ax.set_ylabel("PC2", fontsize=12)
        self.ax.set_zlabel("PC3", fontsize=12)
        self.ax.tick_params(axis='x', which='both', labelsize=12)
        self.ax.tick_params(axis='y', which='both', labelsize=12)
        self.ax.tick_params(axis='z', which='both', labelsize=12)

        # --- lock view, limits, aspect, projection ---

        # Fixed camera orientation you want
        self._elev = 30    # up/down tilt
        self._azim = -60   # left/right rotation
        self._roll = 0     # try 0; on older Matplotlib this is ignored below

        # Your preferred fixed limits
        self._xlim = (-1.0, 1.5)
        self._ylim = (-1.75, 1.25)
        self._zlim = (-0.75, 1.5)
        self.ax.set_xlim(*self._xlim)
        self.ax.set_ylim(*self._ylim)
        self.ax.set_zlim(*self._zlim)

        # Disable autoscale so artists don't move the box
        self.ax.set_autoscale_on(False)

        # Fix camera once
        try:
            self.ax.view_init(elev=self._elev, azim=self._azim, roll=self._roll)  # Matplotlib ≥3.7
        except TypeError:
            self._roll = None
            self.ax.view_init(elev=self._elev, azim=self._azim)

        # Orthographic projection removes perspective “breathing”
        try:
            self.ax.set_proj_type("ortho")
        except Exception:
            pass

        # Keep aspect fixed so scales don't morph
        try:
            self.ax.set_box_aspect((
                self._xlim[1]-self._xlim[0],
                self._ylim[1]-self._ylim[0],
                self._zlim[1]-self._zlim[0],
            ))
        except Exception:
            pass

        (self.line_pre,)  = self.ax.plot([], [], [], color="navy",        lw=2.5, label="Pre-push")
        (self.line_post,) = self.ax.plot([], [], [], color="red",  lw=1.5, label="Post-push")
        self._legend = self.ax.legend(loc="best", fontsize=10, frameon=True, framealpha=0.8)
        (self.head_dot,) = self.ax.plot([], [], [], "o", ms=6, color="#222222", label="Current")
        self._legend = self.ax.legend(loc="best", fontsize=10, frameon=True, framealpha=0.8)

        # An optional scatter for the 3D push point
        self._push_scatter = None

        # Keep last PCA to give stable axes in between recomputes when short
        self._pca = PCA(n_components=3)
        self._pca_fitted = False
        self.min_fit_samples = 50 

        # Nice view box defaults (can be adjusted)
        self.ax.set_xlim(-2, 3.0); self.ax.set_ylim(-2.5, 2.5); self.ax.set_zlim(-1.5, 3.0)

    def set_push_index(self, idx: int):
        self.push_idx = max(0, int(idx))

        # If PCA not yet frozen, fit it on *pre-push* data to lock orientation
        if not self._pca_fitted and len(self.hidden_hist) >= 2:
            H = np.asarray(self.hidden_hist, dtype=np.float32)
            T, D = H.shape if H.ndim == 2 else (0, 0)
            fit_T = min(self.push_idx, T)
            if fit_T >= 2 and D >= 2:
                k = int(min(3, D))
                self._pca = PCA(n_components=k).fit(H[:fit_T])
                self._pca_fitted = True
                # Drop any previous push marker (will be redrawn)
                self._push_scatter = None

        # If we already have points, place/update the 3D push marker
        if len(self.hidden_hist) > self.push_idx:
            self._update_push_scatter()
        self.fig.canvas.draw_idle()


    def _update_push_scatter(self):
        if self.push_idx is None: 
            return
        H = np.asarray(self.hidden_hist, dtype=np.float32)
        if H.ndim != 2 or self.push_idx >= H.shape[0]:
            return
        # Project with current PCA
        try:
            hidden_pca = self._pca.transform(H)
        except Exception:
            hidden_pca = self._pca.fit_transform(H)
        px, py, pz = hidden_pca[self.push_idx]
        if self._push_scatter is None:
            self._push_scatter = self.ax.scatter(px, py, pz, color='red', s=60, label=f"Push (t={self.push_idx})")
            self.ax.legend(loc="best", fontsize=10, frameon=True, framealpha=0.8)
        else:
            self._push_scatter._offsets3d = ([px], [py], [pz])

    def update(self, h_t, step=None, decimate=1):
        """Call this every control step. Matches your existing update signature."""
        if step is None:
            step = self._step_idx
        self._step_idx += 1
        self.hidden_hist.append(np.asarray(h_t, dtype=np.float32).ravel())

        # Throttle redraws
        if (step % max(1, decimate)) != 0:
            return

        H = np.asarray(self.hidden_hist, dtype=np.float32)  # [T, D]
        if H.ndim != 2:
            return

        T, D = H.shape

        if not hasattr(self, "_locked_view"):
            # Capture current interactive angle once, then reuse forever
            self._elev, self._azim = self.ax.elev, self.ax.azim
            self._roll = getattr(self.ax, "roll", self._roll)
            self._locked_view = True


        # How many dims do we *have*?
        k_max = int(min(3, D))

        # If PCA isn’t frozen yet, decide when to fit:
        if not self._pca_fitted:
            # Prefer to lock at push time using pre-push history
            if self.push_idx is not None:
                fit_T = min(self.push_idx, T)
                if fit_T >= 2 and k_max >= 2:
                    self._pca = PCA(n_components=k_max).fit(H[:fit_T])
                    self._pca_fitted = True
                else:
                    # Not enough data yet to lock → wait
                    self.fig.canvas.draw_idle(); plt.pause(0.001); return
            else:
                # No push yet: lock after min_fit_samples (or when enough data)
                if T >= self.min_fit_samples and k_max >= 2:
                    self._pca = PCA(n_components=k_max).fit(H[:self.min_fit_samples])
                    self._pca_fitted = True
                else:
                    # Still gathering pre-fit → optionally show just the latest point
                    self.fig.canvas.draw_idle(); plt.pause(0.001); return

        # From here on, PCA is *frozen*: only transform
        try:
            Pk = self._pca.transform(H)   # [T, k]
        except Exception:
            # Safety: if something went odd, refit once on what we have and lock
            self._pca = PCA(n_components=k_max).fit(H[:max(2, min(T, self.min_fit_samples))])
            self._pca_fitted = True
            Pk = self._pca.transform(H)

        # Pad to 3D if needed
        if Pk.shape[1] < 3:
            P = np.pad(Pk, ((0, 0), (0, 3 - Pk.shape[1])), mode="constant")
        else:
            P = Pk
        self.head_dot.set_data_3d([P[-1,0]], [P[-1,1]], [P[-1,2]])
        # Decide split index (pre vs post)
        if self.push_idx is None:
            split = P.shape[0]
        else:
            split = int(np.clip(self.push_idx, 0, P.shape[0]))

        # Pre-/post- slices
        Ppre  = P[:split]
        Ppost = P[split:]

        # Update the two lines (note: set_data_3d is the stable way)
        if Ppre.shape[0] >= 1:
            self.line_pre.set_data_3d(Ppre[:, 0], Ppre[:, 1], Ppre[:, 2])
        else:
            self.line_pre.set_data_3d([], [], [])

        if Ppost.shape[0] >= 1:
            self.line_post.set_data_3d(Ppost[:, 0], Ppost[:, 1], Ppost[:, 2])
        else:
            self.line_post.set_data_3d([], [], [])

        # Push marker (optional red dot)
        if self.push_idx is not None:
            self._update_push_scatter()

        self.fig.canvas.draw_idle()
        plt.pause(0.001)





def push_recovery_test(robot_name, urdf_path, push_forces, directions, checkpoint=10000,
                       push_duration=0.1, pre_push_steps=50, post_push_steps=200, num_trials=3, pre_push_condition=None):
    """
    Evaluate standing push recovery over multiple trials.
    """
    from legged_gym.utils import get_args, task_registry
    from terrain import Terrain as CustomTerrain
    import legged_gym.utils.terrain as terrain_module
    import types

    print("🌟 Starting push recovery test...")

    results = []
    ts_all_rows = []
    trial_uid = 0 


    for force in push_forces:
        for direction in directions:
            for trial in range(num_trials):
                print(f"\n💥 Trial {trial+1}/{num_trials}: Preparing {force}N push towards {direction}")

                args.headless = False
                args.task = "h1"

                env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
                env_cfg.domain_rand.push_robots = False

                if args.robot == "H1_var":
                    env_cfg.control.variable_stiffness = True
                    env_cfg.control.fixed_stiffness = False
                    env_cfg.control.stiffness_model = STIFFNESS_MODEL
                elif args.robot == "H1_soft":
                    env_cfg.control.variable_stiffness = False
                    env_cfg.control.fixed_stiffness = True
                    env_cfg.control.stiffness_model = STIFFNESS_MODEL
                    env_cfg.control.fixed_stiffness_values = {
                        'stiffness': [200.0, 100.0, 80.0, 200.0, 100.0] * 2,
                        'damping':   [4.5, 4.5, 1.0, 5.0, 4.0] * 2
                    }
                elif args.robot == "H1_normal":
                    env_cfg.control.variable_stiffness = False
                    env_cfg.control.fixed_stiffness = True
                    env_cfg.control.stiffness_model = STIFFNESS_MODEL
                    env_cfg.control.fixed_stiffness_values = {
                        'stiffness': [0.0] * 10,
                        'damping':   [0.0] * 10
                    }
                else:
                    raise ValueError("Unknown robot type. Use H1_normal, H1_soft, or H1_var.")

                train_cfg.runner.resume = True
                train_cfg.runner.load_run = robot_name
                train_cfg.runner.checkpoint = checkpoint

                env_cfg.terrain.selected = True
                env_cfg.terrain.curriculum = False
                env_cfg.terrain.randomize = False
                env_cfg.env.num_envs = 1

                if urdf_path:
                    os.environ["URDF_OVERRIDE_PATH"] = urdf_path

                terrain_module.Terrain = CustomTerrain
                env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
                env.custom_origins = True
                env.env_origins[:] = torch.tensor([[0, 0, 0.0]], device='cuda')


                foot_idx = []
                if hasattr(env, "left_foot_indices") and hasattr(env, "right_foot_indices"):
                    # preferred: use your env's foot index lists
                    foot_idx = list(set(list(env.left_foot_indices) + list(env.right_foot_indices)))
                else:
                    # fallback: find bodies named like feet/toes by name
                    foot_idx = []
                    for b in range(env.num_bodies):
                        name = env.gym.get_actor_rigid_body_name(env.envs[0], env.actor_handles[0], b)
                        if ("foot" in name.lower()) or ("toe" in name.lower()):
                            foot_idx.append(b)

                all_idx = list(range(env.num_bodies))
                nonfoot_idx = [i for i in all_idx if i not in set(foot_idx)]
                env.nonfoot_idx = torch.tensor(nonfoot_idx, device=env.device, dtype=torch.long)

                env.apply_external_force = types.MethodType(apply_external_force, env)

                ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
                actor_critic = ppo_runner.alg.actor_critic

                # Safe expected length for the colorbar normalization
                pre_n  = int(pre_push_steps) if isinstance(pre_push_steps, (int, np.integer)) else 0
                post_n = int(post_push_steps) if isinstance(post_push_steps, (int, np.integer)) else 0
                exp_steps = pre_n + post_n
                if exp_steps <= 0:
                    exp_steps = 1000  # conservative default if both unknown

                lstm_plot = OnlineLSTMAttractor3D(
                    title="LSTM attractor (PCA, live)",
                    expect_steps=exp_steps,
                    decim_draw=2
                )


                lstm_hidden_history = []   # one entry per timestep
                lstm_cell_history  = []    # None entries for GRU to keep alignment

                # === Now do the push test for this trial
                env.reset()
                warmup(env, actor_critic, num_steps=50)
                correct_lateral_velocity(env)

                obs, _ = env.get_observations()
                env.commands[:] = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=env.device)

                com_positions = []
                zmp_positions = []
                start_pos = env.root_states[:, 0:3].clone().cpu().numpy()
                pre_com_positions = []

                # --- PRE-PUSH: either time-locked (legacy) or condition-locked (phase/contacts) ---
                if pre_push_condition is None:
                    # legacy: fixed number of steps
                    for _ in range(pre_push_steps):
                        actions = actor_critic.act_inference(obs)
                        _record_rnn_state(actor_critic, lstm_hidden_history, lstm_cell_history)

                        # Extract a 1-D vector from the first layer, env 0
                        _hidden_states, _ = actor_critic.get_hidden_states()
                        if isinstance(_hidden_states, tuple):  # LSTM: (hx, cx)
                            _hx = _hidden_states[0]
                            h_vec = _hx[0, 0, :].detach().cpu().numpy()
                        else:  # GRU
                            _hx = _hidden_states
                            h_vec = (_hx[0, 0, :].detach().cpu().numpy()
                                    if _hx.dim() == 3 else _hx[0, :].detach().cpu().numpy())

                        # Use total steps so far as x-axis (pre-push count works fine)
                        lstm_plot.update(h_vec, step=len(lstm_hidden_history), decimate=2)


                        obs, _, _, _, _ = env.step(actions.detach())
                        env.maybe_clear_forces()
                        env.render()
                        _update_camera(env, top_view_offset=1.0)
                        try:
                            com_x, com_y, _ = env._compute_com()
                            pre_com_positions.append([float(com_x.mean().item()), float(com_y.mean().item())])
                        except Exception:
                            base_xy = env.root_states[0, 0:2].tolist()
                            pre_com_positions.append([float(base_xy[0]), float(base_xy[1])])
                else:
                    # new: step until condition(env) says "ready"
                    # (keeps the exact same step body as above)
                    while True:
                        actions = actor_critic.act_inference(obs)
                        _record_rnn_state(actor_critic, lstm_hidden_history, lstm_cell_history)

                        # Extract a 1-D vector from the first layer, env 0
                        _hidden_states, _ = actor_critic.get_hidden_states()
                        if isinstance(_hidden_states, tuple):  # LSTM: (hx, cx)
                            _hx = _hidden_states[0]
                            h_vec = _hx[0, 0, :].detach().cpu().numpy()
                        else:  # GRU
                            _hx = _hidden_states
                            h_vec = (_hx[0, 0, :].detach().cpu().numpy()
                                    if _hx.dim() == 3 else _hx[0, :].detach().cpu().numpy())

                        # Use total steps so far as x-axis (pre-push count works fine)
                        lstm_plot.update(h_vec, step=len(lstm_hidden_history), decimate=2)

                        obs, _, _, _, _ = env.step(actions.detach())
                        env.maybe_clear_forces()
                        env.render()
                        _update_camera(env, top_view_offset=1.0)
                        try:
                            com_x, com_y, _ = env._compute_com()
                            pre_com_positions.append([float(com_x.mean().item()), float(com_y.mean().item())])
                        except Exception:
                            base_xy = env.root_states[0, 0:2].tolist()
                            pre_com_positions.append([float(base_xy[0]), float(base_xy[1])])
                        # check phase/contact condition here
                        if pre_push_condition(env):
                            break



                import numpy as _np
                BASELINE_WINDOW = 50  # last N steps of pre-push to average
                if len(pre_com_positions) > 0:
                    baseline_slice = pre_com_positions[-min(BASELINE_WINDOW, len(pre_com_positions)):]
                    baseline_com = _np.mean(_np.array(baseline_slice), axis=0).tolist()  # [x_mean, y_mean]
                else:
                    baseline_com = env.root_states[0, 0:2].detach().cpu().numpy().tolist()

                # Build a force in ROBOT frame
                f_local = torch.zeros_like(env.root_states[:, 0:3])
                if direction == "front":   f_local[:, 0] =  force
                elif direction == "back":  f_local[:, 0] = -force
                elif direction == "left":  f_local[:, 1] =  force
                elif direction == "right": f_local[:, 1] = -force

                # Rotate by current yaw into WORLD frame
                yaw = env.rpy[0, 2].item()
                c, s = math.cos(yaw), math.sin(yaw)
                rot_matrix = torch.tensor([[c, -s, 0],
                                [s,  c, 0],
                                [0,  0, 1]], device=env.device, dtype=f_local.dtype)
                f_world = (f_local @ rot_matrix.T)

                # Apply (torso/base is fine)
                env.apply_external_force(f_world, duration=push_duration)
                push_start_iter = len(pre_com_positions)  # for phase-locked path; equals pre_push_steps in legacy path
                push_start_time_s = push_start_iter * float(env.dt) # dt = 0.01
                print(f"🔹 Push started at global step {push_start_iter} (t = {push_start_time_s:.3f}s)")
                print(f"💥 Push applied: {force}N {direction}")

                if lstm_plot is not None:
                    lstm_plot.set_push_index(push_start_iter)

                trial_idx = trial  # keep the per-combo index if you like
                this_trial_uid = trial_uid
                trial_uid += 1


                # ==== Draw arrow (use f_world, not the old force_vector) ====
                start_point = env.root_states[:, 0:3][0].cpu().numpy()
                force_dir   = f_world[0].detach().cpu().numpy()  # <— fix here
                force_dir_norm = force_dir / (np.linalg.norm(force_dir) + 1e-6)
                arrow_length = 0.5
                end_point   = start_point
                start_point = end_point - arrow_length * force_dir_norm

                env.gym.add_lines(
                    env.viewer, env.envs[0], 1,
                    np.array([[start_point[0], start_point[1], start_point[2],
                            end_point[0],   end_point[1],   end_point[2]]], dtype=np.float32),
                    np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
                )

                # Arrowhead
                arrow_dir      = force_dir_norm
                orthogonal_1   = np.array([-arrow_dir[1], arrow_dir[0], 0])
                arrowhead_len  = 0.2
                head1 = end_point - arrowhead_len * (arrow_dir - 0.5 * orthogonal_1)
                head2 = end_point - arrowhead_len * (arrow_dir + 0.5 * orthogonal_1)
                env.gym.add_lines(
                    env.viewer, env.envs[0], 2,
                    np.array([[end_point[0], end_point[1], end_point[2], head1[0], head1[1], head1[2]],
                            [end_point[0], end_point[1], end_point[2], head2[0], head2[1], head2[2]]], dtype=np.float32),
                    np.array([[1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0]], dtype=np.float32)
                )
                env.pushing = True

                # --- thresholds (unchanged) ---
                TH_COM_SPEED = 0.25          # m/s threshold for stability
                GRACE_SEC    = push_duration + 0.20
                GRACE_N      = max(1, int(GRACE_SEC / env.dt))
                ENTER_STABLE_SEC = 0.30      # need this many seconds below threshold
                ENTER_STABLE_N   = max(1, int(ENTER_STABLE_SEC / env.dt))
                COMS_EMA_ALPHA = 0.3
                com_speed_ema = None
                # --- Walking steps counting (from push -> recovery) ---
                CONTACT_N_TH = 30.0               # contact force threshold [N]
                MIN_STRIDE_SEC = 0.25             # minimum time between counted steps
                MIN_STRIDE_STEPS = max(1, int(MIN_STRIDE_SEC / env.dt))
                gait_steps = 0
                gait_steps_at_recovery = None
                prev_left_contact = False
                prev_right_contact = False
                last_step_foot = None             # "L" or "R"
                last_step_idx  = -MIN_STRIDE_STEPS


                # For finite-difference CoM velocity
                prev_cx_for_speed, prev_cy_for_speed = None, None

                # --- Track deviations ---
                peak_com_dev = 0.0
                dev_at_recovery = None
                dev_last = 0.0

                # --- Walking steps counting ---
                gait_steps = 0
                gait_steps_at_recovery = None


                recovered = False
                fell = False
                recovery_steps = None
                fell_link = None
                fell_step = None
                stable_counter = 0
                ts_log = []

                for step_idx in range(post_push_steps):
                    if getattr(env, "pushing", False):
                        env.gym.apply_rigid_body_force_tensors(
                            env.sim, gymtorch.unwrap_tensor(env.force_tensor), None, gymapi.ENV_SPACE
                        )

                    actions = actor_critic.act_inference(obs)
                    _record_rnn_state(actor_critic, lstm_hidden_history, lstm_cell_history)

                    _hidden_states, _ = actor_critic.get_hidden_states()
                    if isinstance(_hidden_states, tuple):  # LSTM
                        _hx = _hidden_states[0]
                        h_vec = _hx[0, 0, :].detach().cpu().numpy()
                    else:  # GRU
                        _hx = _hidden_states
                        h_vec = (_hx[0, 0, :].detach().cpu().numpy()
                                if _hx.dim() == 3 else _hx[0, :].detach().cpu().numpy())

                    lstm_plot.update(h_vec, step=len(lstm_hidden_history), decimate=2)


                    obs, _, _, dones, _ = env.step(actions.detach())

                    env.maybe_clear_forces()
                    env.render()
                    _update_camera(env, top_view_offset=1.0)
                    

                    lin_vel = env.root_states[:, 7:10]
                    speed = torch.norm(lin_vel[0, :2]).item()

                    roll  = env.rpy[0, 0].item()
                    pitch = env.rpy[0, 1].item()
                    base_z = env.root_states[0, 2].item()


                    # ---------- 2b) FALL = any non-foot link touching ground (with grace) ----------
                    ROOT_FALL_Z_TH = 0.30  # meters

                    if (not fell) and (base_z < ROOT_FALL_Z_TH):
                        fell = True
                        fell_link = "base"
                        fell_step = step_idx
                        print("ROBOT FELT")


                    # --- CoM (always) ---
                    cx = cy = None
                    try:
                        com_x, com_y, _ = env._compute_com()
                        cx = float(com_x.mean().item())
                        cy = float(com_y.mean().item())
                    except Exception:
                        base_xy = env.root_states[0, 0:2].tolist()
                        cx, cy = float(base_xy[0]), float(base_xy[1])

                    # --- CoM planar speed (finite-difference) ---
                    if prev_cx_for_speed is None:
                        com_speed = 0.0
                    else:
                        dx = cx - prev_cx_for_speed
                        dy = cy - prev_cy_for_speed
                        com_speed = (dx*dx + dy*dy) ** 0.5 / env.dt
                    prev_cx_for_speed, prev_cy_for_speed = cx, cy

                    # Optional EMA smoothing
                    com_speed_ema = com_speed if (com_speed_ema is None) else (COMS_EMA_ALPHA * com_speed + (1.0 - COMS_EMA_ALPHA) * com_speed_ema)

                    # --- Count consecutive "stable" frames after a small grace period ---
                    speed_ok = (com_speed_ema if com_speed_ema is not None else com_speed) < TH_COM_SPEED
                    if step_idx >= GRACE_N:
                        stable_counter = stable_counter + 1 if speed_ok else 0
                    else:
                        stable_counter = 0  # don't count during grace

                    # --- keep a log of CoM for stats ---
                    com_positions.append([cx, cy])

                    # --- CoM deviation from pre-push baseline (for dev_* metrics) ---
                    dxb = cx - float(baseline_com[0])
                    dyb = cy - float(baseline_com[1])
                    dev = (dxb * dxb + dyb * dyb) ** 0.5

                    # track last and peak deviation
                    dev_last = dev
                    if dev > peak_com_dev:
                        peak_com_dev = dev

                    # ----- FOOT CONTACTS (rising edges = steps) -----
                    try:
                        left_z  = env.contact_forces[0, env.left_foot_indices,  2].max().item()
                        right_z = env.contact_forces[0, env.right_foot_indices, 2].max().item()
                    except Exception:
                        left_z, right_z = 0.0, 0.0

                    left_contact  = (left_z  > CONTACT_N_TH)
                    right_contact = (right_z > CONTACT_N_TH)

                    counted = False
                    if left_contact and not prev_left_contact:
                        if (step_idx - last_step_idx) >= MIN_STRIDE_STEPS and last_step_foot != "L":
                            gait_steps += 1
                            last_step_foot = "L"
                            last_step_idx  = step_idx
                            counted = True

                    if (not counted) and right_contact and not prev_right_contact:
                        if (step_idx - last_step_idx) >= MIN_STRIDE_STEPS and last_step_foot != "R":
                            gait_steps += 1
                            last_step_foot = "R"
                            last_step_idx  = step_idx

                    prev_left_contact, prev_right_contact = left_contact, right_contact


                    # --- Latch recovery: ONLY the CoM-speed stability criterion ---
                    if (not recovered) and (stable_counter >= ENTER_STABLE_N):
                        recovered = True
                        recovery_steps = step_idx      # steps since the push ended
                        dev_at_recovery = dev          # keep your existing dev tracking
                        gait_steps_at_recovery = gait_steps
                        print(f"✅ Recovery after {recovery_steps} steps (≈ {recovery_steps * env.dt:.3f}s). CoM speed ~ {(com_speed_ema if com_speed_ema is not None else com_speed):.3f} m/s")

                    ts_log.append({
                        "trial_uid": this_trial_uid,     # <-- unique across whole run
                        "trial_idx": trial_idx,          # <-- 0..num_trials-1 within a combo
                        "force": force,
                        "direction": direction,
                        "step_idx": step_idx,
                        "time_s": step_idx * env.dt,
                        "com_x": cx,
                        "com_y": cy,
                        "com_speed": com_speed_ema if com_speed_ema is not None else com_speed,
                        "dev_from_baseline": dev,
                        "fell": int(fell),
                        "recovered": int(recovered),
                        "left_contact_force": float(left_z),
                        "right_contact_force": float(right_z),
                    })

                # ---- Save RNN histories for this trial ----
                log_data = {
                    "meta": {
                        "trial_uid": this_trial_uid,
                        "trial_idx": trial_idx,
                        "force": force,
                        "direction": direction,
                        "dt": float(env.dt),

                        # robust to phase-locked runs
                        "pre_push_steps": int(pre_push_steps) if pre_push_steps is not None else -1,
                        "post_push_steps": int(post_push_steps),

                        # exact timing you computed at force application
                        "push_start_iter": int(push_start_iter),
                        "push_start_time_s": float(push_start_time_s),

                        # (optional) add trigger info for bookkeeping
                        "trigger_mode": "phase" if pre_push_condition is not None else "fixed_steps",
                        "trigger_support_foot": getattr(pre_push_condition, "support_foot", None) if pre_push_condition is not None else None,
                        "trigger_phase_target": float(getattr(pre_push_condition, "target", -1.0)) if pre_push_condition is not None else None,
                        "trigger_phase_tol": float(getattr(pre_push_condition, "tol", -1.0)) if pre_push_condition is not None else None,
                    },
                    "lstm_hidden_history": lstm_hidden_history,
                    "lstm_cell_history": lstm_cell_history,
                }


                # If you want one file per trial, build a unique path. Otherwise use json_path as-is.
                trial_json_path = json_path  # or e.g. f"{json_path.rstrip('.json')}_trial{this_trial_uid}.json"

                tmp_path = trial_json_path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(log_data, f, indent=2)
                os.rename(tmp_path, trial_json_path)
                print(f"✅ LSTM log saved to {trial_json_path}")

                # Gracefully close the live plot for this trial
                try:
                    import matplotlib.pyplot as _plt
                    _plt.pause(0.001)
                    _plt.close(lstm_plot.fig)
                except Exception:
                    pass


                # Remove push arrow
                if hasattr(env, "pushing") and env.pushing:
                    env.gym.clear_lines(env.viewer)
                    env.pushing = False


                end_pos = env.root_states[:, 0:3].clone().cpu().numpy()
                displacement = np.linalg.norm(end_pos[0][:2] - start_pos[0][:2])
                success_final = int(not fell)


                # Time since push (s) only if we actually recovered
                rec_time_s_out = float(recovery_steps * env.dt) if recovered else -1.0
                rec_steps_out  = int(recovery_steps)            if recovered else -1
                gait_steps_out = int(gait_steps_at_recovery)    if (recovered and gait_steps_at_recovery is not None) else -1


                # CoM deviation at the moment of recovery (or last value if never recovered)
                dev_final = dev_at_recovery if (dev_at_recovery is not None) else dev_last
                dev_peak = peak_com_dev

                # ---- after the for step_idx loop, before results.append({...}) ----
                pre_std = np.std(np.array(pre_com_positions), axis=0).tolist() if pre_com_positions else [0.0, 0.0]
                post_std = np.std(np.array(com_positions), axis=0).tolist() if com_positions else [0.0, 0.0]
                # Convert ts_log to a DataFrame so we can add constant columns easily
                ts_df = pd.DataFrame(ts_log)
                # Add clear push timing flags (separate push vs grace)
                push_steps = max(1, int(push_duration / env.dt))
                ts_df["push_on"]  = (ts_df["step_idx"] < push_steps).astype(int)
                ts_df["grace_on"] = ((ts_df["step_idx"] >= push_steps) & (ts_df["step_idx"] < GRACE_N)).astype(int)
                # Add CoM std dev (pre/post) as constants for this trial
                ts_df["com_std_pre_x"]  = pre_std[0]
                ts_df["com_std_pre_y"]  = pre_std[1]
                ts_df["com_std_post_x"] = post_std[0]
                ts_df["com_std_post_y"] = post_std[1]
                # (Optional) also include recovery annotations as constants
                ts_df["recovery_steps"]   = rec_steps_out
                ts_df["recovery_time_s"]  = rec_time_s_out
                ts_df["recovered_flag"]   = int(recovered)
                ts_df["trial_uid"]   = this_trial_uid
                ts_df["force"]       = force
                ts_df["direction"]   = direction
                # Accumulate into the one big list
                ts_all_rows.extend(ts_df.to_dict(orient="records"))



                results.append({
                    "trial_uid": this_trial_uid,
                    "force": force,
                    "direction": direction,
                    "success": float(success_final),
                    "fell": int(fell),
                    "recovery_steps": rec_steps_out,
                    "recovery_time_s": rec_time_s_out,
                    "recovery_walking_steps": gait_steps_out,
                    "recovery_disp_final_m": dev_final,
                    "recovery_disp_peak_m": dev_peak,
                    "displacement": displacement,
                })

                # ⚡ DESTROY the Isaac Gym simulation for this trial
                env.gym.destroy_sim(env.sim)
                if hasattr(env, "viewer") and env.viewer is not None:
                    env.gym.destroy_viewer(env.viewer)

                del env
                del ppo_runner
                del actor_critic
                torch.cuda.empty_cache()


    # Write the one big time-series CSV
    ts_all_df = pd.DataFrame(ts_all_rows)
    ts_all_path = "legged_gym/plots/timeseries_all_trials.csv"
    ts_all_df.to_csv(ts_all_path, index=False)
    print(f"📈 Unified time-series log written: {ts_all_path}")

    # Return the summary DataFrame (one row per trial)
    return pd.DataFrame(results)





def apply_external_force(self, force_tensor, duration=1.0,
                         body_preference=("chest","torso","upper_body","upperbody","trunk","waist","base")):
    env_h = self.envs[0]
    act_h = self.actor_handles[0]

    # Try preferred names first
    target_body_idx = -1
    target_body_name = None
    for name in body_preference:
        idx = self.gym.find_actor_rigid_body_handle(env_h, act_h, name)
        if idx != -1:
            target_body_idx = idx
            target_body_name = name
            break

    # Fallbacks if none matched (avoid UnboundLocalError)
    if target_body_idx == -1:
        # Try the actor's root body (index 0) as a safe default
        target_body_idx = 0
        target_body_name = "body_0"  # label for logging

    # one-time init: (num_bodies, 3)
    if not hasattr(self, 'force_tensor'):
        self.force_tensor = torch.zeros((self.num_bodies, 3), device=self.device)

    # Zero then set the selected body force (world frame)
    self.force_tensor[:] = 0.0
    # force_tensor is shape (num_envs=1, 3) → take [0]
    self.force_tensor[target_body_idx] = force_tensor[0]

    # Apply now; while `self.pushing` is True we re-apply each step
    self.gym.apply_rigid_body_force_tensors(
        self.sim, gymtorch.unwrap_tensor(self.force_tensor), None, gymapi.ENV_SPACE
    )

    # Timed clearing: let env.maybe_clear_forces() stop it
    import time
    self.pushing = True
    self.push_end_time = time.time() + float(duration)

    # Save for arrow drawing / debug
    self.last_push_body_idx = int(target_body_idx)
    self.last_push_body_name = str(target_body_name)

    print(f"[push] applying {force_tensor[0].tolist()} to body '{target_body_name}' (idx={target_body_idx}) for {duration}s")




def maybe_clear_forces(self):
    # Stop applying external forces once the timer (set in apply_external_force) expires
    if getattr(self, "pushing", False):
        import time
        if time.time() >= getattr(self, "push_end_time", 0.0):
            if hasattr(self, "force_tensor"):
                self.force_tensor[:] = 0.0
                self.gym.apply_rigid_body_force_tensors(
                    self.sim,
                    gymtorch.unwrap_tensor(self.force_tensor),
                    None,
                    gymapi.ENV_SPACE
                )
            self.pushing = False

























def play(args, terrain_type, custom_terrain_kwargs=None):
    print(f"\n🌍 Evaluating terrain: {terrain_type}")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.domain_rand.push_robots = False

    if args.robot == "H1_var":
        env_cfg.control.variable_stiffness = True
        env_cfg.control.fixed_stiffness = False
        env_cfg.control.stiffness_model = STIFFNESS_MODEL
    elif args.robot == "H1_soft":
        env_cfg.control.variable_stiffness = False
        env_cfg.control.fixed_stiffness = True
        env_cfg.control.stiffness_model = STIFFNESS_MODEL
        env_cfg.control.fixed_stiffness_values = {
            'stiffness': [200.0, 100.0, 80.0, 200.0, 100.0] * 2,
            'damping':   [4.5, 4.5, 1.0, 5.0, 4.0] * 2
        }
    elif args.robot == "H1_normal":
        env_cfg.control.variable_stiffness = False
        env_cfg.control.fixed_stiffness = True
        env_cfg.control.stiffness_model = STIFFNESS_MODEL
        env_cfg.control.fixed_stiffness_values = {
            'stiffness': [0.0] * 10,
            'damping':   [0.0] * 10
        }
    else:
        raise ValueError("Unknown robot type. Use H1_normal, H1_soft, or H1_var.")

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = LOAD_RUN
    train_cfg.runner.checkpoint = CHECKPOINT

    env_cfg.terrain.selected = True
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.randomize = False
    env_cfg.env.num_envs = 1  
    env_cfg.terrain.terrain_length = 8.0
    env_cfg.terrain.terrain_width = 8.0
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 10

    if "stepping_stones" in terrain_type:
        env_cfg.terrain.terrain_kwargs = custom_terrain_kwargs or {
            "type": "terrain_utils.stepping_stones_terrain",
            "stone_size": 0.25,
            "stone_height": 0.05,
            "spacing": 0.4,
            "num_rows": 50,
            "num_cols": 50,
        }
    elif "discrete_obstacles" in terrain_type:
        env_cfg.terrain.terrain_kwargs = custom_terrain_kwargs or {
            "type": "custom_discrete_obstacles",
            "num_rects": 15,
            "min_size": 0.2,
            "max_size": 0.6,
            "max_height": 0.08
        }
    elif "stairs" in terrain_type:
        env_cfg.terrain.terrain_kwargs = custom_terrain_kwargs or {
            "type": "custom_stairs",
            "num_steps": 10, # Total number of stairs.
            "step_height": 0.05, # Height of each step (in meters).
            "step_length": 0.4, # Length of each step (in meters) — how much forward it goes.
            "step_width": 2.0 # Width of the step (in y-axis direction).
        }
    elif "wave" in terrain_type:
        env_cfg.terrain.terrain_kwargs = custom_terrain_kwargs or {
            "type": "custom_wave", 
            "num_waves": 5, # Number of full sine wave cycles.
            "amplitude": 0.05, # Height of the wave (peak-to-center, not peak-to-peak)
            "spacing": 0.4, # Distance between cubes.
            "width": 2.0 # How wide the wave pattern is in the y-axis (how far left/right).
        }
    elif "pyramid" in terrain_type:
        env_cfg.terrain.terrain_kwargs = custom_terrain_kwargs or {
            "type": "custom_pyramid",
            "slope": 0.05, # Height change per layer (affects how tall the pyramid gets).
            "size": 5.0, # Total size (both x and y) of the base of the pyramid.
            "step": 0.4 # Thickness (in meters) of each layer.
        }
    elif "pyramid_obstacles" in terrain_type:
        env_cfg.terrain.terrain_kwargs = custom_terrain_kwargs or {
            "type": "custom_pyramid_obstacles",  # ✅ Must match the type expected in create_sim()
            "num_pyramids": 100,
            "base_size": 0.6,
            "step_height": 0.02,
            "layers": 3,
            "area": 5.0
        }
    elif "spheres" in terrain_type:
        env_cfg.terrain.terrain_kwargs = custom_terrain_kwargs or {
            "type": "custom_spheres", 
            "num_spheres": 150, # Number of spherical obstacles to spawn. Each one will have a random radius and be randomly positioned in the area.
            "min_radius": 0.02, # The minimum radius (in meters) that a sphere can have.
            "max_radius": 0.02, # The maximum radius (in meters) that a sphere can have.
            "area": 5.0 # The square area side length (in meters) where obstacles can appear.
        }
    elif "random_uniform" in terrain_type:
        env_cfg.terrain.terrain_kwargs = custom_terrain_kwargs or {
            "type": terrain_type,
            "min_height": -0.06,
            "max_height": 0.06,
            "step": 0.03
        }
    elif "slope" in terrain_type:
        env_cfg.terrain.terrain_kwargs = custom_terrain_kwargs or {
            "type": "custom_slope",
            "slope": 0.1,
            "size": 2.0,
            "slope_length": 15.0,
            "slope_width": 15.0
        }
    elif terrain_type in ["none", "plane"]:
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.terrain_kwargs = {
            "type": terrain_type
        }
    else:
        raise ValueError(f"Unknown terrain type: {terrain_type}")



    env_cfg.terrain.mesh_type = "patch"


    terrain_module.Terrain = CustomTerrain
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.custom_origins = True
    # env.env_origins[:] = torch.tensor([[0.5, 0.5, 0.2]], device='cuda') # robot position : for slope
    env.env_origins[:] = torch.tensor([[-0.5, 0.5, 0.2]], device='cuda')


    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    actor_critic = ppo_runner.alg.actor_critic

    joint_names = env.dof_names 
    joint_energy = {name: 0.0 for name in joint_names}
    joint_max_vel = {name: 0.0 for name in joint_names}
    joint_max_acc = {name: 0.0 for name in joint_names}
    joint_max_pos = {name: 0.0 for name in joint_names}
    joint_spring_return = {name: 0.0 for name in joint_names}
    joint_damper_return = {name: 0.0 for name in joint_names}
    energy_tot = {name: 0.0 for name in joint_names}
    energy_tot_per_joint = {name: 0.0 for name in joint_names}
    max_contact_force_z_left = 0.0
    max_contact_force_z_right = 0.0
    
    results = {}

    for (vx, vy, yaw, label) in tasks:
        print(f"\n🚶 Task: {label}")
        results[label] = {"success": [], "energy": [], "max_joint_vel": [], "com_zmp_var": [], "steps": []}

        for trial in range(num_trials):
            env.reset()
            warmup(env, actor_critic, num_steps=50)
            correct_lateral_velocity(env)

             # 🛠 Reset per-trial measurements
            joint_energy = {name: 0.0 for name in joint_names}
            joint_max_vel = {name: 0.0 for name in joint_names}
            joint_max_acc = {name: 0.0 for name in joint_names}
            joint_max_pos = {name: 0.0 for name in joint_names}
            max_contact_force_z_left = 0.0
            max_contact_force_z_right = 0.0

            start_pos = env.root_states[:, 0:3].clone().cpu().numpy()
            # time.sleep(999999)  # Pause so you can inspect

            video_out_path = os.path.join(plots_dir, f"video_{terrain_type}_{label}_{trial}.mp4")

            if RECORD_FRAMES:
                # Create camera sensors
                frame_width = 3840
                frame_height = 2160
                cam_props = gymapi.CameraProperties()
                cam_props.width = frame_width
                cam_props.height = frame_height
                cam_handle = env.gym.create_camera_sensor(env.envs[0], cam_props)

                cam2_props = gymapi.CameraProperties()
                cam2_props.width = frame_width
                cam2_props.height = frame_height
                cam2_handle = env.gym.create_camera_sensor(env.envs[0], cam2_props)

                # Attach / set camera
                com_pos = env.get_com() if hasattr(env, "get_com") else env.base_pos
                robot_pos = com_pos[0].tolist()

                if CAMERA_VIEW == "BACK":
                    fixed_cam_position = gymapi.Vec3(robot_pos[0] - 0.5, robot_pos[1], robot_pos[2] + 1)
                    roll, pitch, yaw = [0,30,0]
                elif CAMERA_VIEW == "LEFT":
                    fixed_cam_position = gymapi.Vec3(robot_pos[0] + 3.0, robot_pos[1] + 6.0, robot_pos[2] + 1)
                    roll, pitch, yaw = [0,30,-90]
                    # fixed_cam_position = gymapi.Vec3(robot_pos[0] + 1.0, robot_pos[1] + 5.0, robot_pos[2] + 1) # for slope
                    # roll, pitch, yaw = [0,10,-70] # for slope
                elif CAMERA_VIEW == "RIGHT":
                    fixed_cam_position = gymapi.Vec3(robot_pos[0], robot_pos[1] - 3.0, robot_pos[2] + 1)
                    roll, pitch, yaw = [0,30,90]
                elif CAMERA_VIEW == "FRONT":
                    fixed_cam_position = gymapi.Vec3(robot_pos[0] + 5.0, robot_pos[1], robot_pos[2] + 1)
                    roll, pitch, yaw = [0,30,180]
                elif CAMERA_VIEW == "TOP":
                    fixed_cam_position = gymapi.Vec3(robot_pos[0], robot_pos[1], robot_pos[2] + 3.0)
                    roll, pitch, yaw = [0,90,0]

                cam_quaternion = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_quat()
                cam_rotation = gymapi.Quat(cam_quaternion[0], cam_quaternion[1], cam_quaternion[2], cam_quaternion[3])
                cam_transform = gymapi.Transform()
                cam_transform.p = fixed_cam_position
                cam_transform.r = cam_rotation
                env.gym.set_camera_transform(cam2_handle, env.envs[0], cam_transform)

                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = 30
                video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (cam2_props.width, cam2_props.height))

            obs, _ = env.get_observations()
            energy = 0.0
            max_joint_vel = 0.0
            com_zmp_diffs = []
            com_positions = []
            zmp_positions = []
            if "turn_left" in label or "turn_right" in label:
                desired_heading = None  
                correct_lateral = False
            elif "left" in label or "right" in label:
                desired_heading = None  
                correct_lateral = True
            else:
                desired_heading = 0.0  
                correct_lateral = False

            env.commands[:] = torch.tensor([[vx, vy, yaw, 0.0]], device=env.device).expand(env.num_envs, -1)

            if EXPORT_POLICY:
                path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
                export_policy_as_jit(ppo_runner.alg.actor_critic, path)

            lstm_hidden_history = []
            lstm_cell_history = []

            # ⏳ Simulation loop
            start_time = time.time()  # Start timing

            step_idx = 0
            iteration_limit = 100  # or any number of steps you want
            iteration_counter = 0

            while True:
                current_time = time.time()
                if desired_heading is not None:
                    corrected_yaw_rate = adjust_yaw_command(desired_heading, env)
                else:
                    corrected_yaw_rate = yaw

                env.commands[:] = torch.tensor([[vx, vy, corrected_yaw_rate, 0.0]], device=env.device)
                    
                actions = actor_critic.act_inference(obs)

                lin_vel = env.root_states[:, 7:10]
                ang_vel = env.root_states[:, 10:13]

                if correct_lateral:
                    lin_vel[:, 0] *= 0.0  # Zero vx (forward) drift
                    ang_vel[:, 2] *= 0.0  # Zero yaw rotation drift
                else:
                    lin_vel[:, 1] *= 0.9  # Smooth lateral velocity reduction
                    ang_vel[:, 2] *= 0.9  # Smooth yaw rotation reduction
                    target_vx = vx  # ← use the vx from the task
                    lin_vel[:, 0] = 0.9 * lin_vel[:, 0] + 0.1 * target_vx
                    min_vx = 0.2 * np.sign(target_vx) if target_vx != 0 else 0.0
                    lin_vel[:, 0] = torch.clamp(lin_vel[:, 0], min=min_vx)

                env.root_states[:, 7:10] = lin_vel
                env.root_states[:, 10:13] = ang_vel
                gym_root_states_raw = env.gym.acquire_actor_root_state_tensor(env.sim)
                gym_root_states = gymtorch.wrap_tensor(gym_root_states_raw).view(-1, 13)
                gym_root_states.copy_(env.root_states)
                env.gym.set_actor_root_state_tensor(env.sim, gym_root_states_raw)

                # Step the simulation
                obs, _, rews, dones, infos = env.step(actions.detach())

                end_pos = env.root_states[:, 0:3].clone().cpu().numpy()
                displacement_vector = end_pos[0][:2] - start_pos[0][:2]
                displacement = np.linalg.norm(displacement_vector)

                yaw = env.rpy[0][2].item()

                if displacement >= target_distance:
                    print(f"✅ Target distance {target_distance:.1f} m reached.")
                    break
                if current_time - start_time >= max_duration:
                    print(f"⏳ Max simulation time {max_duration:.1f} seconds reached.")
                    break
                if BREAK_WHEN_FALL:
                    if dones.any():
                        print("💥 Robot fell or episode ended.")
                        break

                # ENERGY
                # if iteration_counter < iteration_limit:
                torques = (actions * env.cfg.control.action_scale).detach().cpu().numpy()[0]
                dof_vels = env.dof_vel.detach().cpu().numpy()[0]
                dof_poss = env.dof_pos.detach().cpu().numpy()[0]
                if args.robot == "H1_soft":
                    stiffness = np.array(env.cfg.control.fixed_stiffness_values["stiffness"])
                    damping = np.array(env.cfg.control.fixed_stiffness_values["damping"])
                elif args.robot == "H1_var":
                    full_action = actions.detach().cpu().numpy()[0]
                    torques = full_action[:10] * env.cfg.control.action_scale
                    stiffness_half = full_action[10:15]  # 5 values
                    damping_half = full_action[15:20]    # 5 values
                    # Mirror to get 10 values (assuming order: [L_hip, L_knee, ..., R_hip, R_knee, ...])
                    stiffness = np.concatenate([stiffness_half, stiffness_half])
                    damping = np.concatenate([damping_half, damping_half])
                else:  # H1_normal (rigid)
                    stiffness = np.zeros_like(dof_poss)
                    damping = np.zeros_like(dof_vels)
                prev_dof_vels = env.last_dof_vel[0].cpu().numpy()  # Get previous joint velocities (1st env)
                for i, name in enumerate(joint_names):
                    θ = dof_poss[i]             # Current joint position
                    θ_dot = dof_vels[i]         # Current joint velocity
                    τ = torques[i]              # Torque applied
                    k = stiffness[i]            # Current stiffness
                    d = damping[i]              # Current damping
                    dt = env.dt                 # Time step (more robust than hardcoding 0.005)

                    # Energies
                    motor_energy = max(0.0, τ * θ_dot * dt)
                    spring_energy = 0.5 * k * θ**2
                    damper_energy = d * θ_dot**2 * dt

                    # Logging per-joint metrics
                    joint_energy[name]         += motor_energy
                    joint_spring_return[name]  += spring_energy
                    joint_damper_return[name]  += damper_energy
                    energy_tot_per_joint[name] = joint_energy[name] - joint_damper_return[name]

                    # Max values
                    joint_max_vel[name] = max(joint_max_vel[name], abs(θ_dot))
                    joint_acc = (θ_dot - prev_dof_vels[i]) / dt
                    joint_max_acc[name] = max(joint_max_acc[name], abs(joint_acc))
                    joint_max_pos[name] = max(joint_max_pos[name], abs(θ))

                motor_total = sum(joint_energy.values())
                spring_total = sum(joint_spring_return.values())
                damper_total = sum(joint_damper_return.values())
                energy_tot = motor_total - damper_total

                # Compute CoT after episode ends
                actor_handle = env.actor_handles[0]  # assuming one actor per env
                mass_props = env.gym.get_actor_rigid_body_properties(env.envs[0], actor_handle)
                robot_mass = sum([body.mass for body in mass_props])
                g = 9.81
                cot = motor_total / (robot_mass * g * displacement) if displacement > 0 else float('inf')
                results[label].setdefault("cot", []).append(cot)

                # Feet forces
                left_z = env.contact_forces[:, env.left_foot_indices, 2].max(dim=1).values
                right_z = env.contact_forces[:, env.right_foot_indices, 2].max(dim=1).values
                max_contact_force_z_left = max(max_contact_force_z_left, abs(left_z.max().item()))
                max_contact_force_z_right = max(max_contact_force_z_right, abs(right_z.max().item()))


                com_x, com_y, _ = env._compute_com()
                zmp_x, zmp_y, _, _ = env._compute_zmp()
                if zmp_x is not None and zmp_y is not None:
                    com_x_np = com_x.cpu().numpy()[0]
                    com_y_np = com_y.cpu().numpy()[0]
                    zmp_x_np = zmp_x.cpu().numpy()[0]
                    zmp_y_np = zmp_y.cpu().numpy()[0]
                    com_zmp_dist = np.sqrt((com_x_np - zmp_x_np)**2 + (com_y_np - zmp_y_np)**2)
                    com_zmp_diffs.append(com_zmp_dist)
                    com_positions.append([com_x_np, com_y_np])
                    zmp_positions.append([zmp_x_np, zmp_y_np])


                # Collect data **before** taking the step
                feet_contact = env.feet_state.cpu().numpy() if hasattr(env, "feet_state") else env.contact_forces.cpu().numpy()
                actions_scaled = actions * env.cfg.control.action_scale  # Apply scaling factor
                motor_torque = actions_scaled.detach().cpu().numpy() # Commanded torques (sent by the RL policy, desired torques output by the learned policy)
                sensor_torque = env.torques.detach().cpu().numpy()
                
                # Store in history
                motor_torque_history.append(motor_torque.tolist())
                sensor_torque_history.append(sensor_torque.tolist())
                feet_contact_history.append(feet_contact.tolist())
                joint_angles = env.dof_pos.cpu().numpy() if hasattr(env, "dof_pos") else np.zeros((env.num_envs, env.num_dof))
                joint_angle_history.append(joint_angles.tolist())


                # 🔍 Access LSTM hidden states after action inference
                hidden_states, _ = actor_critic.get_hidden_states()
                if isinstance(hidden_states, tuple):  # LSTM
                    hx, cx = hidden_states
                    hx_np = hx[0].detach().cpu().numpy()  # [layer, batch, dim]
                    cx_np = cx[0].detach().cpu().numpy()
                    lstm_hidden_history.append(hx_np.tolist())
                    lstm_cell_history.append(cx_np.tolist())
                else:  # GRU or simple RNN
                    hx = hidden_states
                    hx_np = hx[0].detach().cpu().numpy()
                    lstm_hidden_history.append(hx_np.tolist())
                    print("✅ GRU Hidden Step:", hx_np.shape)

                if MOVE_CAMERA or RECORD_FRAMES:
                    com_pos = env.root_states[:, 0:3].detach().cpu().numpy()  # Extract CoM position
                    robot_pos = com_pos[0].tolist()
                    camera_position = gymapi.Vec3(robot_pos[0] - 2.0, robot_pos[1], robot_pos[2] + 1.0)
                    camera_target = gymapi.Vec3(robot_pos[0], robot_pos[1], robot_pos[2])
                    cam_transform = gymapi.Transform()
                    cam_transform.p = camera_position  
                    env.gym.viewer_camera_look_at(env.viewer, None, camera_position, camera_target)
                    if hasattr(env, 'cam_handle') and env.cam_handle is not None:
                        env.gym.set_camera_transform(env.cam_handle, env.envs[0], cam_transform)

                if RECORD_FRAMES:
                    env.gym.render_all_camera_sensors(env.sim)
                    frame = env.gym.get_camera_image(env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR)
                    if frame is not None:
                        rgba = np.frombuffer(frame, dtype=np.uint8).reshape(frame_height, frame_width, 4)
                        bgr  = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
                        video_writer.write(bgr)

             
            step_count = len(com_zmp_diffs) 

            if RECORD_FRAMES and video_writer is not None:
                video_writer.release()
                print(f"📹 Video saved to {video_out_path}")


            success = displacement >= target_distance
            com_zmp_var = np.var(com_zmp_diffs) if len(com_zmp_diffs) > 0 else 0.0

            for name in joint_energy:
                if step_count > 0:
                    joint_energy[name] /= step_count  # ➡️ Divide by real number of steps
            for name in joint_names:
                if step_count > 0:
                    joint_energy[name] /= step_count
                results[label][f"energy_{name}"] = results[label].get(f"energy_{name}", [])
                results[label][f"energy_{name}"].append(joint_energy[name])

                results[label][f"max_vel_{name}"] = results[label].get(f"max_vel_{name}", [])
                results[label][f"max_vel_{name}"].append(joint_max_vel[name])

                results[label][f"max_acc_{name}"] = results[label].get(f"max_acc_{name}", [])
                results[label][f"max_acc_{name}"].append(joint_max_acc[name])

                results[label][f"max_pos_{name}"] = results[label].get(f"max_pos_{name}", [])
                results[label][f"max_pos_{name}"].append(joint_max_pos[name])

                results[label][f"max_pos_{name}"] = results[label].get(f"max_pos_{name}", [])
                results[label][f"max_pos_{name}"].append(joint_max_pos[name])

                


            com_array = np.array(com_positions)
            zmp_array = np.array(zmp_positions)

            if com_array.shape[0] > 0:
                results[label]["com_mean"] = results[label].get("com_mean", [])
                results[label]["com_std"] = results[label].get("com_std", [])
                results[label]["com_mean"].append(com_array.mean(axis=0).tolist())
                results[label]["com_std"].append(com_array.std(axis=0).tolist())

            if zmp_array.shape[0] > 0:
                results[label]["zmp_mean"] = results[label].get("zmp_mean", [])
                results[label]["zmp_std"] = results[label].get("zmp_std", [])
                results[label]["zmp_mean"].append(zmp_array.mean(axis=0).tolist())
                results[label]["zmp_std"].append(zmp_array.std(axis=0).tolist())

            results[label]["com_zmp_dist_mean"] = results[label].get("com_zmp_dist_mean", [])
            results[label]["com_zmp_dist_var"] = results[label].get("com_zmp_dist_var", [])
            results[label]["com_zmp_dist_mean"].append(float(np.mean(com_zmp_diffs)))
            results[label]["com_zmp_dist_var"].append(float(np.var(com_zmp_diffs)))
            # Save full time series
            results[label].setdefault("com_full", []).append(com_positions)
            results[label].setdefault("zmp_full", []).append(zmp_positions)

            results[label]["success"].append(float(success))
            results[label].setdefault("motor_energy_total", []).append(motor_total)
            results[label].setdefault("spring_energy_total", []).append(spring_total)
            results[label].setdefault("damper_energy_total", []).append(damper_total)
            results[label].setdefault("net_energy_total", []).append(energy_tot)
            results[label].setdefault("energy", []).append(energy_tot) 

            # Optionally store per-joint energy if needed
            for name in joint_names:
                results[label].setdefault(f"spring_energy_{name}", []).append(joint_spring_return[name])
                results[label].setdefault(f"damper_energy_{name}", []).append(joint_damper_return[name])

            # Other metrics
            max_joint_vel = max(joint_max_vel.values())
            max_joint_acc = max(joint_max_acc.values())
            max_joint_pos = max(joint_max_pos.values())
            results[label].setdefault("max_joint_vel", []).append(max_joint_vel)
            results[label].setdefault("max_joint_acc", []).append(max_joint_acc)
            results[label].setdefault("max_joint_pos", []).append(max_joint_pos)
            results[label].setdefault("com_zmp_var", []).append(com_zmp_var)
            results[label].setdefault("distance", []).append(float(displacement))
            results[label].setdefault("steps", []).append(step_count)
            results[label].setdefault("cot", []).append(cot)
            results[label].setdefault("max_contact_force_z_left", []).append(max_contact_force_z_left)
            results[label].setdefault("max_contact_force_z_right", []).append(max_contact_force_z_right)
            results[label].setdefault("max_contact_force_z", []).append(max(max_contact_force_z_left, max_contact_force_z_right))


            # ✅ Clear, aligned print statements
            print(f"✅ Trial {trial+1}: Success = {success}, Distance = {displacement:.2f} m")
            print(f"🔋 Motor Energy Total: {motor_total:.2f}")
            print(f"🌀 Spring Energy Stored: {spring_total:.2f}")
            print(f"🌊 Damper Energy Dissipated: {damper_total:.2f}")
            print(f"💡 Net Energy (Motor - Damper): {energy_tot:.2f}")
            print(f"⚡ Max Joint Velocity: {max_joint_vel:.2f}")
            print(f"📈 CoM-ZMP Var: {com_zmp_var:.4f}")
            print(f"🦶 Steps Taken: {step_count}")

            

    print(f"🧹 Cleaning up simulator...")
    env.gym.destroy_sim(env.sim)
    if hasattr(env, "viewer") and env.viewer is not None:
        env.gym.destroy_viewer(env.viewer)

    # 🧹 Delete Python objects
    del env
    del ppo_runner
    del actor_critic
    torch.cuda.empty_cache()  # Optional: clean GPU memory

    print(f"✅ Simulator and objects destroyed.")


    return results






if __name__ == '__main__':
    import argparse
    import sys

    # First, extract custom args (robot, checkpoint)
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["H1_normal", "H1_soft", "H1_var"], required=True)
    parser.add_argument("--checkpoint", type=int, default=10000)
    parser.add_argument("--stiffness_model", choices=["parallel", "series"], default="parallel")
    parser.add_argument("--load_run", type=str, default=None)
    args_config, unknown = parser.parse_known_args()

    # Remove custom args from sys.argv before get_args()
    sys.argv = [sys.argv[0]] + unknown

    # Now import and call get_args()
    from legged_gym.utils import get_args
    args = get_args()

    # Inject the custom values
    args.robot = args_config.robot
    CHECKPOINT = args_config.checkpoint
    LOAD_RUN = args_config.load_run if args_config.load_run is not None else args_config.robot
    STIFFNESS_MODEL = args_config.stiffness_model

    urdf_path = None

    num_trials = 1 # How many trials for each task
    max_duration = 10000000  # Stop after 30 seconds
    target_distance = 10 # 8.0

    tasks = [
        (0.8, 0.0, 0.0, "forward"),
        (-0.8, 0.0, 0.0, "backward"),
        (0.0, 0.6, 0.0, "left"),
        (0.0, -0.6, 0.0, "right"),
        (0.4, 0.0, 0.8, "forward_turn_left"),
        (0.4, 0.0, -0.8, "forward_turn_right")
    ]

    terrain_types = [
        "stairs_4dir",
        "plane",
        "pyramid_obstacles",
        "discrete_obstacles",
        "stepping_stones",
        "spheres",
        "custom_slope",
    ]

    # Define obstacles height to test
    height_values = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]
    slope_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65]


    terrain_param_templates = {
        "stepping_stones": {
            "type": "terrain_utils.stepping_stones_terrain",
            "stone_size": 0.08,
            "spacing": 0.26,
            "num_rows": 200,
            "num_cols": 80
        },
        "discrete_obstacles": {
            "type": "custom_discrete_obstacles",
            "num_rects": 25000,
            "area": 30,
            "min_size": 0.01,
            "max_size": 0.1,
            "max_height": None
        },
        "spheres": {
            "type": "custom_spheres",
            "num_spheres": 12000,
            "min_radius": 0.03,
            "max_radius": 0.05,
            "area": 30.0
        },
        "stairs": {
            "type": "custom_stairs",
            "num_steps": 40,
            "step_length": 0.5,
            "step_width": 5.0,
            "step_height": None
        },
        "pyramid_obstacles": {
            "type": "custom_pyramid_obstacles", 
            "num_pyramids": 17000,
            "base_size": 0.07,
            "step_height": None,
            "layers": 3,
            "area": 30.0
        },
        "plane": {
            "type": "plane"
        },
        "custom_slope": {
            "type": "custom_slope",
            "slope": None,
            "size": 2.0,
            "slope_length": 15.0,
            "slope_width": 15.0
        },
        "stairs_4dir": {
            "type": "custom_stairs_4dir",
            "num_steps": 40,
            "step_length": 0.7,
            "step_width": 3.0,
            "step_height": None,
            "center_size": 1.5  # Optional: size of the flat central platform
        },
    }


    # ✨ Uneven terrains test
    all_dfs = []

    for terrain_type in terrain_types:
        if terrain_type == "flat":
            params = terrain_param_templates[terrain_type].copy()
            print(f"\n🚀 Evaluating Terrain: {terrain_type}")
            
            df = evaluate_policy(
                args=args,
                load_run=LOAD_RUN,
                urdf_override=urdf_path,
                terrains=[terrain_type],
                custom_terrain_kwargs=params
            )
            df["stone_height"] = 0.0
            all_dfs.append(df)

        elif terrain_type in ["slope", "custom_slope"]:
            for slope in slope_values:
                params = terrain_param_templates["custom_slope"].copy()
                params["slope"] = float(slope)
                print(f"\n🚀 Evaluating Terrain: {terrain_type} with slope {slope:.2f}")
                df = evaluate_policy(
                    args=args,
                    load_run=LOAD_RUN,
                    urdf_override=urdf_path,
                    terrains=[terrain_type],              # stays "custom_slope"
                    custom_terrain_kwargs=params          # now contains a real slope
                )
                df["stone_height"] = 0.0
                df["slope"] = slope
                all_dfs.append(df)  
        

        else:
            for height in height_values:
                params = terrain_param_templates[terrain_type].copy()
                if terrain_type == "stepping_stones":
                    params["stone_height"] = height
                elif terrain_type == "discrete_obstacles":
                    params["max_height"] = height
                elif terrain_type == "spheres":
                    params["min_radius"] = height - 0.02
                    params["max_radius"] = height - 0.02
                elif terrain_type == "stairs":
                    params["step_height"] = height
                elif terrain_type == "stairs_4dir":
                    params["step_height"] = height
                elif terrain_type == "pyramid_obstacles":
                    params["step_height"] = height/params["layers"]

                print(f"\n🚀 Evaluating Terrain: {terrain_type} with height {height:.2f}")
                
                df = evaluate_policy(
                    args=args,
                    load_run=LOAD_RUN,
                    urdf_override=urdf_path,
                    terrains=[terrain_type],
                    custom_terrain_kwargs=params
                )
                df["stone_height"] = height
                all_dfs.append(df)


    # ✅ Save all results together
    final_df = pd.concat(all_dfs, ignore_index=True)
    save_path = f"legged_gym/plots/eval_results_{args.robot}_allterrains.csv"
    final_df.to_csv(save_path, index=False)
    print(f"\n✅ All terrains evaluated and saved into {save_path}")


    # ✨ Push recovery test
    push_forces = [760, 770, 780, 790, 800] 
    directions = ["left", "right","back", "front"]
    # Push arrive at mid-stance (50% of the stance of the “support” foot)
    phase_trigger = PhaseLockedTrigger(
        support_foot="left",   # or "right"
        target=0.50,           # mid-stance
        tol=0.02,              # ±2% phase window
        warmup_steps=150,      # allow cadence to settle
        contact_N=30.0,        # same as your step counter threshold
        hold_frames=3
    )

    push_results = push_recovery_test(
        robot_name=LOAD_RUN,
        urdf_path=urdf_path,
        push_forces=push_forces,
        directions=directions,
        checkpoint=CHECKPOINT,
        push_duration=0.5,
        pre_push_steps=None,             # disable fixed time
        post_push_steps=400,
        num_trials=num_trials,
        pre_push_condition=phase_trigger # <-- activate phase-locked trigger
    )

    save_path = f"legged_gym/plots/push_recovery_{args.robot}.csv"
    push_results.to_csv(save_path, index=False)
    print(f"✅ Push recovery tests saved to {save_path}")