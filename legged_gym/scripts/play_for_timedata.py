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
import csv


EXPORT_POLICY = False
RECORD_FRAMES = False
MOVE_CAMERA = True

CAMERA_VIEW = "LEFT"  # Options: "BACK", "FRONT", "RIGHT", "LEFT", "TOP"

plots_dir = "legged_gym/plots"
video_out_path_hq = os.path.join(plots_dir, "simulation_video_HQ.mp4")

# === Logging control ===
LOG_STRIDE = 1   # sample every N sim steps to lighten GPU load
SUMMARY_CSV = "legged_gym/plots/data/h1_timedata.csv"
TIMESERIES_CSV = "legged_gym/plots/data/h1_timeseries.csv"



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



def push_recovery_test(robot_name, urdf_path, push_forces, directions, checkpoint=10000,
                       push_duration=0.1, pre_push_steps=50, post_push_steps=200, num_trials=3):
    """
    Evaluate standing push recovery over multiple trials.
    """
    from legged_gym.utils import get_args, task_registry
    from terrain import Terrain as CustomTerrain
    import legged_gym.utils.terrain as terrain_module
    import types

    print("🌟 Starting push recovery test...")

    results = []

    for force in push_forces:
        for direction in directions:
            for trial in range(num_trials):
                print(f"\n💥 Trial {trial+1}/{num_trials}: Preparing {force}N push towards {direction}")

                # ⚡ CREATE a fresh Isaac Gym simulation
                # orig_argv = sys.argv.copy()
                # sys.argv = [sys.argv[0]]
                # args = get_args()
                # sys.argv = orig_argv

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

                env.apply_external_force = types.MethodType(apply_external_force, env)

                ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
                actor_critic = ppo_runner.alg.actor_critic

                # === Now do the push test for this trial
                env.reset()
                warmup(env, actor_critic, num_steps=50)
                correct_lateral_velocity(env)

                obs, _ = env.get_observations()
                env.commands[:] = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=env.device)

                com_positions = []
                zmp_positions = []
                start_pos = env.root_states[:, 0:3].clone().cpu().numpy()

                for _ in range(pre_push_steps):
                    actions = actor_critic.act_inference(obs)
                    obs, _, _, _, _ = env.step(actions.detach())
                    env.maybe_clear_forces()
                    env.render()
                    _update_camera(env, top_view_offset=1.0)

                # Apply external push
                force_vector = torch.zeros_like(env.root_states[:, 0:3])
                if direction == "front":
                    force_vector[:, 0] = -force
                elif direction == "back":
                    force_vector[:, 0] = force
                elif direction == "left":
                    force_vector[:, 1] = force
                elif direction == "right":
                    force_vector[:, 1] = -force

                force_tensor = force_vector.to(env.device)
                env.apply_external_force(force_tensor, duration=push_duration)
                print(f"💥 Push applied: {force}N {direction}")

                # Draw arrow
                start_point = env.root_states[:, 0:3][0].cpu().numpy()
                force_dir = force_vector[0].cpu().numpy()
                force_dir_norm = force_dir / (np.linalg.norm(force_dir) + 1e-6)
                arrow_length = 0.5
                end_point = start_point
                start_point = end_point - arrow_length * force_dir_norm
                env.gym.add_lines(
                    env.viewer,
                    env.envs[0],
                    1,
                    np.array([[start_point[0], start_point[1], start_point[2],
                            end_point[0], end_point[1], end_point[2]]], dtype=np.float32),
                    np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
                )
                arrow_dir = force_dir_norm
                orthogonal_1 = np.array([-arrow_dir[1], arrow_dir[0], 0])  # rotate 90° in XY plane
                orthogonal_2 = np.cross(arrow_dir, [0, 0, 1])              # alternative orthogonal

                # Shorten and scale for arrowhead arms
                arrowhead_length = 0.2
                head1 = end_point - arrowhead_length * (arrow_dir - 0.5 * orthogonal_1)
                head2 = end_point - arrowhead_length * (arrow_dir + 0.5 * orthogonal_1)

                env.gym.add_lines(
                    env.viewer,
                    env.envs[0],
                    2,
                    np.array([
                        [end_point[0], end_point[1], end_point[2], head1[0], head1[1], head1[2]],
                        [end_point[0], end_point[1], end_point[2], head2[0], head2[1], head2[2]],
                    ], dtype=np.float32),
                    np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
                )
                env.pushing = True

                # Post-push
                recovered = False
                recovery_steps = None

                for step_idx in range(post_push_steps):
                    if getattr(env, "pushing", False):
                        env.gym.apply_rigid_body_force_tensors(
                            env.sim,
                            gymtorch.unwrap_tensor(env.force_tensor),
                            None,
                            gymapi.ENV_SPACE
                        )

                    actions = actor_critic.act_inference(obs)
                    obs, _, _, dones, _ = env.step(actions.detach())
                    env.maybe_clear_forces()
                    env.render()
                    _update_camera(env, top_view_offset=1.0)

                    lin_vel = env.root_states[:, 7:10]
                    speed = torch.norm(lin_vel[0, :2]).item()

                    if speed < 0.1 and not recovered:
                        recovery_steps = step_idx
                        recovered = True
                        print(f"✅ Recovery detected after {recovery_steps} steps.")

                    if env.root_states[0, 2].item() < 0.25:
                        print("💥 Base too low, robot likely fell.")
                        recovered = False
                        break

                    com_x, com_y, _ = env._compute_com()
                    zmp_x, zmp_y, _, _ = env._compute_zmp()
                    if zmp_x is not None and zmp_y is not None:
                        com_positions.append([com_x.mean().item(), com_y.mean().item()])
                        zmp_positions.append([zmp_x.mean().item(), zmp_y.mean().item()])

                # Remove push arrow
                if hasattr(env, "pushing") and env.pushing:
                    env.gym.clear_lines(env.viewer)
                    env.pushing = False

                end_pos = env.root_states[:, 0:3].clone().cpu().numpy()
                displacement = np.linalg.norm(end_pos[0][:2] - start_pos[0][:2])

                results.append({
                    "trial": trial,
                    "force": force,
                    "direction": direction,
                    "success": float(recovered),
                    "recovery_steps": recovery_steps if recovery_steps is not None else post_push_steps,
                    "displacement": displacement,
                    "com_std": np.std(np.array(com_positions), axis=0).tolist(),
                    "zmp_std": np.std(np.array(zmp_positions), axis=0).tolist()
                })

                # ⚡ DESTROY the Isaac Gym simulation for this trial
                env.gym.destroy_sim(env.sim)
                if hasattr(env, "viewer") and env.viewer is not None:
                    env.gym.destroy_viewer(env.viewer)

                del env
                del ppo_runner
                del actor_critic
                pass

    return pd.DataFrame(results)





def apply_external_force(self, force_tensor, duration=0.1):
    """ Apply external force for a number of simulation steps. """
    env_handle = self.envs[0]
    actor_handle = self.actor_handles[0]

    if not hasattr(self, 'base_body_idx'):
        self.base_body_idx = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, "base")

    base_body_idx = self.base_body_idx

    if not hasattr(self, 'force_tensor'):
        self.force_tensor = torch.zeros((self.num_envs * self.num_bodies, 3), device=self.device)

    self.force_tensor[:] = 0.0
    self.force_tensor[base_body_idx] = force_tensor[0]

    self.gym.apply_rigid_body_force_tensors(
        self.sim,
        gymtorch.unwrap_tensor(self.force_tensor),
        None,
        gymapi.ENV_SPACE
    )

    # 💡 NEW: set end step count instead of wall time
    self.pushing = True
    self.push_remaining_steps = int(duration * 1.0 / self.dt)  # e.g. duration=0.5s, dt=0.005 ⇒ 100 steps


def maybe_clear_forces(self):
    if getattr(self, "pushing", False):
        self.push_remaining_steps -= 1
        if self.push_remaining_steps <= 0:
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

    log_time = []
    log_com = []
    log_zmp = []
    log_foot_forces = []
    log_joint_pos = []
    log_joint_acc = []
    log_joint_vel = []

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
    # env.env_origins[:] = torch.tensor([[-1, 0, 0.2]], device='cuda')  # Robot position -15.7 (stairs -1)
    env.env_origins[:] = torch.tensor([[-0.5, 0.5, 0.2]], device='cuda')


    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    actor_critic = ppo_runner.alg.actor_critic

    joint_names = env.dof_names 

    
    results = {}
    summary_rows = []

    for (vx, vy, yaw, label) in tasks:
        print(f"\n🚶 Task: {label}")
        results[label] = {"success": [], "energy": [], "max_joint_vel": [], "com_zmp_var": [], "steps": []}

        for trial in range(num_trials):
            env.reset()
            warmup(env, actor_critic, num_steps=50)
            correct_lateral_velocity(env)

            step_idx = 0
            prev_dof_vel = None
            max_fz_left = 0.0
            max_fz_right = 0.0
            com_xy_samples = []    # for CoM std dev (x, y)
            max_abs_joint_acc = torch.zeros(env.num_dof, device=env.device)
            printed_indices = False
            ts_rows = [] 


            start_pos = env.root_states[:, 0:3].clone().cpu().numpy()
            # time.sleep(999999)  # Pause so you can inspect

            
            obs, _ = env.get_observations()

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


            # ⏳ Simulation loop
            start_time = time.time()  # Start timing


            # ---- Safe foot-force indexing ----
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

                # Show the sim (unless headless)
                if not args.headless:
                    env.render()




                # ---- Light-weight logging every LOG_STRIDE steps ----

                step_idx += 1
                if step_idx % LOG_STRIDE == 0:
                    dt = env.dt

                    # Defaults so we never reference before assignment
                    fz_left_step, fz_right_step = float("nan"), float("nan")
                    acc0 = None


                    # --- SAFE foot indices & vertical force ---
                    L_idx = torch.as_tensor(env.left_foot_indices, device=env.device, dtype=torch.long).view(-1)
                    R_idx = torch.as_tensor(env.right_foot_indices, device=env.device, dtype=torch.long).view(-1)
                    all_idx = torch.cat([L_idx, R_idx], dim=0)

                    num_bodies = env.contact_forces.shape[1]
                    if torch.any(all_idx < 0) or torch.any(all_idx >= num_bodies):
                        print(f"[IndexError] Invalid foot index! num_bodies={num_bodies}, "
                            f"L_idx={L_idx.tolist()}, R_idx={R_idx.tolist()}")
                    else:
                        if not printed_indices:
                            try:
                                names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
                                print("Rigid bodies:", names)
                                print("Left foot idx:", L_idx.tolist(), "->", [names[i] for i in L_idx.tolist()])
                                print("Right foot idx:", R_idx.tolist(), "->", [names[i] for i in R_idx.tolist()])
                            except Exception as e:
                                print(f"(warn) couldn’t fetch rigid body names: {e}")
                                print("Left foot idx:", L_idx.tolist())
                                print("Right foot idx:", R_idx.tolist())
                            printed_indices = True

                        foot_forces = env.contact_forces[:, all_idx, :]  # (N_envs, nBodies, 3)
                        nL = L_idx.numel()
                        fz_left_step  = float(foot_forces[0, :nL, 2].sum().item())
                        fz_right_step = float(foot_forces[0, nL:, 2].sum().item())

                        # keep trial maxima
                        if fz_left_step  > max_fz_left:  max_fz_left  = fz_left_step
                        if fz_right_step > max_fz_right: max_fz_right = fz_right_step


                    # --- CoM sample ---
                    if hasattr(env, "_compute_com"):
                        com_x, com_y, _ = env._compute_com()
                        com_xy_samples.append((float(com_x.mean().item()), float(com_y.mean().item())))
                    elif hasattr(env, "get_com"):
                        c = env.get_com()[0]
                        com_xy_samples.append((float(c[0].item()), float(c[1].item())))
                    else:
                        base_xy = env.root_states[0, 0:2]
                        com_xy_samples.append((float(base_xy[0].item()), float(base_xy[1].item())))

                    # --- Per-joint acceleration ---
                    curr_dof_vel = env.dof_vel.clone()
                    if prev_dof_vel is not None:
                        joint_acc = (curr_dof_vel - prev_dof_vel) / dt      # [num_envs, num_dof]
                        acc0 = torch.abs(joint_acc[0])                       # [num_dof]
                        max_abs_joint_acc = torch.maximum(max_abs_joint_acc, acc0)
                    else:
                        # first sample: no previous velocity → zeros
                        acc0 = torch.zeros(env.num_dof, device=env.device)
                    prev_dof_vel = curr_dof_vel

                    # --- Current per-joint stiffness/damping (10 values) ---
                    # Prefer reading from env (if available), otherwise reconstruct from action or fixed config
                    try:
                        # Some envs expose the *actual* applied stiffness/damping
                        curr_stiff = env.curr_stiffness[0].detach().float().cpu().tolist()  # len = 10
                        curr_damp  = env.curr_damping[0].detach().float().cpu().tolist()     # len = 10
                    except Exception:
                        # Fall back depending on controller mode
                        if getattr(env.cfg.control, "variable_stiffness", False):
                            # Reconstruct from the last 'actions' (same way you did in play_forVideo.py)
                            # NOTE: Only torques are scaled by action_scale; stiffness/damping parts are raw.
                            full_action = actions.detach().cpu().numpy()[0]
                            # Expect layout: [10 torques | 5 stiffness | 5 damping] for one side, mirrored to 10
                            if full_action.shape[0] >= 20:
                                stiff_half = full_action[10:15]
                                damp_half  = full_action[15:20]
                                curr_stiff = np.concatenate([stiff_half, stiff_half]).tolist()
                                curr_damp  = np.concatenate([damp_half,  damp_half]).tolist()
                            else:
                                # Safer fallback if action layout differs
                                curr_stiff = [float("nan")] * env.num_dof
                                curr_damp  = [float("nan")] * env.num_dof
                        else:
                            # Fixed-stiffness robots
                            fs = env.cfg.control.fixed_stiffness_values
                            curr_stiff = list(fs.get("stiffness", [0.0] * env.num_dof))
                            curr_damp  = list(fs.get("damping",   [0.0] * env.num_dof))





                    # --- Build a per-step row for the time series ---
                    timestamp = float(env.episode_length_buf[0].item() * env.dt)

                    row_ts = {
                        "time": timestamp,
                        "robot": LOAD_RUN,
                        "terrain": terrain_type,
                        "task": label,
                        "trial": trial,
                        "fz_left": fz_left_step,
                        "fz_right": fz_right_step,
                        "com_x": float(com_xy_samples[-1][0]) if com_xy_samples else float("nan"),
                        "com_y": float(com_xy_samples[-1][1]) if com_xy_samples else float("nan"),
                    }

                    dof_names = getattr(env, "dof_names", [f"dof_{i}" for i in range(env.num_dof)])
                    acc_vals_step = acc0.detach().float().cpu().tolist()

                    for i, name in enumerate(dof_names):
                        row_ts[f"acc_abs_{name}"] = acc_vals_step[i]
                        row_ts[f"stiff_{name}"]   = float(curr_stiff[i])  # NEW
                        row_ts[f"damp_{name}"]    = float(curr_damp[i])   # NEW

                    ts_rows.append(row_ts)









                # --- Logging metrics ---
                
                dt = env.dt
                timestamp = env.episode_length_buf[0].item() * dt
                log_time.append(timestamp)



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
                if dones.any():
                    print("💥 Robot fell or episode ended.")
                    break

               
            

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


            # === End-of-trial summaries ===
            com_xy_arr = np.array(com_xy_samples) if len(com_xy_samples) else np.zeros((1,2))
            com_std_x, com_std_y = np.std(com_xy_arr, axis=0).tolist()

            # Base row
            row = {
                "robot": LOAD_RUN,
                "terrain": terrain_type,
                "task": label,
                "trial": trial,
                "max_fz_left": max_fz_left,
                "max_fz_right": max_fz_right,
                "max_fz_total": max(max_fz_left, max_fz_right),
                "com_std_x": com_std_x,
                "com_std_y": com_std_y,
            }

            # Per-joint max |acceleration|
            dof_names = getattr(env, "dof_names", [f"dof_{i}" for i in range(env.num_dof)])
            acc_vals = max_abs_joint_acc.detach().float().cpu().tolist()  # 1D list length = num_dof
            for i, name in enumerate(dof_names):
                row[f"max_acc_{name}"] = acc_vals[i]

            # (optional) keep overall scalar too
            row["max_joint_acc_overall"] = float(max_abs_joint_acc.max().item())

            summary_rows.append(row)


            # === Write per-step time series for this trial ===
            os.makedirs(os.path.dirname(TIMESERIES_CSV), exist_ok=True)
            df_ts = pd.DataFrame(ts_rows)
            write_header_ts = not os.path.exists(TIMESERIES_CSV)
            df_ts.to_csv(TIMESERIES_CSV, mode="a", index=False, header=write_header_ts)
            print(f"📝 Appended {len(df_ts)} samples to {TIMESERIES_CSV}")


    # === Save compact per-trial summary CSV ===
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    df_sum = pd.DataFrame(summary_rows)
    # Append (create if missing)
    write_header = not os.path.exists(SUMMARY_CSV)
    df_sum.to_csv(SUMMARY_CSV, mode="a", index=False, header=write_header)
    print(f"✅ Wrote {len(df_sum)} rows to {SUMMARY_CSV}")

   

            

    print(f"🧹 Cleaning up simulator...")
    env.gym.destroy_sim(env.sim)
    if hasattr(env, "viewer") and env.viewer is not None:
        env.gym.destroy_viewer(env.viewer)

    # 🧹 Delete Python objects
    del env
    del ppo_runner
    del actor_critic
    pass

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
    target_distance = 8.0 # 8.0

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
    slope_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


    terrain_param_templates = {
        "stepping_stones": {
            "type": "terrain_utils.stepping_stones_terrain",
            "stone_size": 0.07,
            "spacing": 0.2,
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
            "num_spheres": 200,
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
            "step_length": 0.35,
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



    # ✨ Push recovery test
    push_forces = [200, 300, 400, 500, 600]  # higher forces
    directions = ["left", "right","back", "front"]

    push_results = push_recovery_test(
        robot_name=LOAD_RUN,        
        urdf_path=urdf_path,
        push_forces=push_forces,
        directions=directions,
        checkpoint=CHECKPOINT,
        push_duration=2.0,
        pre_push_steps=100,
        post_push_steps=200,
        num_trials=num_trials
    )



    save_path = f"legged_gym/plots/push_recovery_{args.robot}.csv"
    push_results.to_csv(save_path, index=False)
    print(f"✅ Push recovery tests saved to {save_path}")