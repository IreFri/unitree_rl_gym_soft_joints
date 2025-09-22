


from legged_gym.envs.base.legged_robot import LeggedRobot
import math
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

import time
from warnings import WarningMessage
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import sys
sys.path.append("legged_gym/utils")
from terrain import Terrain  


class H1Robot(LeggedRobot):

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless, args=None):
        self.cfg = cfg

        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.args = args
        self.custom_origins = False 
        self.stiffness_log = []
        self.damping_log = []
        self.max_joint_pos = torch.tensor(0.0)
        self.max_joint_acc = torch.tensor(0.0)
        self.prev_joint_vel = torch.tensor(0.0)
        self.max_contact_force_z_left = torch.tensor(0.0)
        self.max_contact_force_z_right = torch.tensor(0.0)


        self.episode_count = 0

        if self.args is not None and hasattr(self.args, "urdf") and self.args.urdf:
            urdf_filename = self.args.urdf
            urdf_path = os.path.join(
                os.getenv("LEGGED_GYM_ROOT_DIR", "legged_gym"), 
                "../resources/robots/h1/urdf/", 
                urdf_filename
            )
            urdf_path = os.path.abspath(urdf_path)
            self.cfg.asset.file = urdf_path
        # print(f"✅ Using custom URDF from args: {self.cfg.asset.file}")


        self._parse_cfg(self.cfg)

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        # ✅ Logging and plotting setup
        self.plot_update_frequency = 100
        self.reward_history = []
        self.torque_history = {i: [] for i in range(self.num_dof)}  # needs num_dof initialized
        self.csv_path = "legged_gym/scripts/training_logs.csv"
        self.log_data = {
            "step": [],
            "episode": [],
            "joint_id": [],
            "torque": [],
            "action": [],
            "reward": [],
        }

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True


        if self.cfg.control.variable_stiffness:
            self.prev_stiffness = torch.zeros((self.num_envs, 10), device=self.device)




    def create_sim(self):
        """Creates simulation, terrain and environments"""
        self.envs = []
        self.actor_handles = []
        self.robot_indices = []


        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        if not self.headless:
            light_color = gymapi.Vec3(0.3, 0.3, 0.3)
            light_dir = gymapi.Vec3(0.5, 0.5, 0.6)
            ambient = gymapi.Vec3(0.3, 0.3, 0.4)
            self.gym.set_light_parameters(self.sim, 0, light_color, light_dir, ambient)

        
        if self.cfg.terrain.mesh_type == 'plane':
            print("🌱 Using flat ground. Skipping patch terrain setup.")
            self._create_ground_plane()
            self._get_env_origins()           # get one origin per env
            self._create_envs()               # create robot envs

        elif self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            print(f"🌄 Using {self.cfg.terrain.mesh_type} terrain.")
            # ✅ 1. Create terrain using the full number of envs you want
            self.terrain = Terrain(self.cfg.terrain, self.cfg.env.num_envs)
            if self.cfg.terrain.mesh_type == "heightfield":
                self._create_heightfield()
            elif self.cfg.terrain.mesh_type == "trimesh":
                self._create_trimesh()
            self._get_env_origins()
            if self.cfg.terrain.measure_heights:
                self.height_samples = torch.tensor(self.terrain.heightsamples, device=self.device)
            self._create_envs()
        elif self.cfg.terrain.mesh_type == "patch":
            print("🪵 Using patch-based terrain with spawn functions.")
            self._create_ground_plane()       # spawn a flat plane
            self._get_env_origins()           # get origins for each env
            self._create_envs()               # create the envs, fill self.envs

            for i in range(self.num_envs):
                origin = self.env_origins[i]
                self._spawn_patch_terrain(
                    patch_type=self.cfg.terrain.terrain_kwargs['type'],
                    origin=origin,
                    size_x=self.cfg.terrain.terrain_length,
                    size_y=self.cfg.terrain.terrain_width,
                    kwargs=self.cfg.terrain.terrain_kwargs,
                    terrain_env=self.envs[i]        # this is now valid
                )

        else:
            raise ValueError(f"❌ Unsupported mesh_type: {self.cfg.terrain.mesh_type}")
        
        self.robot_indices = torch.tensor(
            [self.gym.get_actor_index(self.envs[i], self.actor_handles[i], gymapi.DOMAIN_SIM)
            for i in range(self.num_envs)],
            dtype=torch.long,
            device=self.device
        )

        # ✅ Check consistency
        assert len(self.envs) == self.cfg.env.num_envs, f"Expected {self.cfg.env.num_envs} envs, got {len(self.envs)}"
        assert len(self.actor_handles) == self.cfg.env.num_envs, f"Expected {self.cfg.env.num_envs} actor handles"


        
 









    def maybe_clear_forces(self):
        """ Clear external forces and visual arrows if push duration expired. """
        if hasattr(self, "push_end_time") and time.time() > self.push_end_time:
            self.force_tensor[:] = 0.0
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.force_tensor),
                None,
                gymapi.ENV_SPACE
            )
            if hasattr(self, "pushing"):
                self.pushing = False
            if hasattr(self, "viewer"):
                self.gym.clear_lines(self.viewer)
            del self.push_end_time


    def post_physics_step_init(self):
        super().post_physics_step_init()
        self.force_tensor = torch.zeros((self.num_bodies, 3), dtype=torch.float32, device=self.device)
        


    def _create_heightfield(self):
        # Get raw heightfield and dimensions
        hf_raw = self.terrain.height_field_raw.astype(np.int16)
        rows, cols = hf_raw.shape

        # Set heightfield parameters
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution
        hf_params.nbRows = rows
        hf_params.nbColumns = cols

        # Flatten heightfield for API
        hf_data = hf_raw.flatten()

        # Register with simulator
        self.gym.add_heightfield(self.sim, hf_data, hf_params)




    def _create_trimesh(self):
        vertices = self.terrain.vertices.astype(np.float32)
        triangles = self.terrain.triangles.astype(np.uint32)

        mesh_asset = self.gym.create_triangle_mesh(
            self.sim,
            vertices.flatten(),
            triangles.flatten(),
            len(vertices),
            len(triangles),
            gymapi.MeshFlags.NONE
        )

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

        # add the mesh **to the world**, not to each env
        self.gym.create_actor(
            self.sim,  # note: put it on the sim, not in each env
            mesh_asset,
            pose,
            "trimesh_terrain",
            0,
            0,
            0
        )





    def _spawn_patch_terrain(self, patch_type, origin, size_x, size_y, kwargs, terrain_env):
        kwargs_clean = kwargs.copy()
        kwargs_clean.pop("type", None)

        if patch_type == "terrain_utils.stepping_stones_terrain":
            self._spawn_stepping_stones(origin, terrain_env, **kwargs_clean)
        elif patch_type == "custom_discrete_obstacles":
            self._spawn_discrete_obstacles(origin, terrain_env, **kwargs_clean)
        elif patch_type == "custom_stairs":
            self._spawn_stairs(origin, terrain_env, **kwargs_clean)
        elif patch_type == "custom_wave":
            self._spawn_wave(origin, terrain_env, **kwargs_clean)
        elif patch_type == "custom_pyramid":
            self._spawn_pyramid_slope(origin, terrain_env, **kwargs_clean)
        elif patch_type == "custom_spheres":
            self._spawn_sphere_obstacles(origin, terrain_env, **kwargs_clean)
        elif patch_type == "custom_pyramid_obstacles":
            self._spawn_pyramid_obstacles(origin, terrain_env, **kwargs_clean)
        elif patch_type == "custom_slope":
            self._spawn_slope(
                origin,
                terrain_env,
                slope=kwargs_clean.get("slope", 0.05),
                size=kwargs_clean.get("size", 2.0),
                slope_length=kwargs_clean.get("slope_length", 15.0),
                slope_width=kwargs_clean.get("slope_width", 15.0)
            )
        elif patch_type == "custom_stairs_4dir":
            self._spawn_stairs_4dir(origin, terrain_env, **kwargs_clean)









    def _get_random_terrain_kwargs(self, patch_type, size_x, size_y):
        # Use local patch size to constrain terrain features
        area = min(size_x, size_y) - 1.0  # Optional margin to avoid overflow

        if patch_type == "custom_stairs":
            step_length = 0.3  # meters
            step_width = size_y * 0.9      # fit inside the patch vertically
            num_steps = int(size_x / step_length)
            return dict(
                num_steps=num_steps,
                step_height=np.random.uniform(0.05, 0.15),
                step_length=step_length,
                step_width=step_width
            )

        elif patch_type == "custom_discrete_obstacles":
            return dict(
                num_rects=np.random.randint(40, 46),
                min_size=0.03,
                max_size=0.06,
                max_height=np.random.uniform(0.01, 0.03),
                area=area
            )

        elif patch_type == "terrain_utils.stepping_stones_terrain":
            spacing = np.random.uniform(0.25, 0.4)  # ensure not too dense
            num_rows = max(1, int(size_x / spacing))
            num_cols = max(1, int(size_y / spacing))
            return dict(
                stone_size=np.random.uniform(0.02, 0.04),
                stone_height=np.random.uniform(0.01, 0.03),
                spacing=spacing,
                num_rows=num_rows,
                num_cols=num_cols
            )

        elif patch_type == "custom_wave":
            return dict(
                num_waves=np.random.randint(15, 25),
                amplitude=np.random.uniform(0.02, 0.05),
                spacing=np.random.uniform(0.6, 1.2),
                width=size_y * 0.9  # wave width constrained to patch
            )

        elif patch_type == "custom_pyramid":
            return dict(
                slope=np.random.uniform(0.2, 0.3),
                size=min(size_x, size_y) * 0.9,  # use most of the patch
                step=np.random.uniform(0.5, 0.8)
            )

        elif patch_type == "custom_spheres":
            return dict(
                num_spheres=np.random.randint(50, 70),
                area=area,
                min_radius=0.01,
                max_radius=0.04
            )
        elif patch_type == "custom_slope":
            # mild slope random between 2% and 8%
            slope = np.random.uniform(0.02, 0.08)
            length = size_x * 0.9
            width = size_y * 0.9
            return dict(
                slope=slope,
                length=length,
                width=width
            )

        elif patch_type == "plane":
            # flat terrain, no parameters
            return {}
        else:
            raise ValueError(f"Unknown patch_type: {patch_type}")

        
    def create_box_asset(self, size_x, size_y, size_z):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        return self.gym.create_box(self.sim, size_x, size_y, size_z, asset_options)

    def _spawn_pyramid_obstacles(self, origin, terrain_env, size_x=None, size_y=None, **kwargs):
        num_pyramids = kwargs.get("num_pyramids", 10)
        base_size = kwargs.get("base_size", 0.3)
        step_height = kwargs.get("step_height", 0.02)
        layers = kwargs.get("layers", 3)
        area = kwargs.get("area", 5.0)

        origin_x = origin[0].cpu().item()
        origin_y = origin[1].cpu().item()
        origin_z = origin[2].cpu().item()

        safe_radius = 0.5

        for i in range(num_pyramids):
            x = np.random.uniform(origin_x - area / 2, origin_x + area / 2)
            y = np.random.uniform(origin_y - area / 2, origin_y + area / 2)
            base_z = origin_z

            if np.linalg.norm([x - origin_x, y - origin_y]) < safe_radius:
                continue

            for l in range(layers):
                size = base_size - l * (base_size / layers)
                height = step_height
                z = base_z + height * (l + 0.5)

                asset = self.create_box_asset(size, size, height)
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(x, y, z)
                self.gym.create_actor(terrain_env, asset, pose, f"pyramid_obs_{i}_layer_{l}", 0, 0)


    def _spawn_stepping_stones(self, origin, terrain_env, stone_size=0.25, stone_height=0.05, spacing=0.4, num_rows=5, num_cols=5):
        box_asset_options = gymapi.AssetOptions()
        box_asset_options.fix_base_link = True
        box_asset = self.gym.create_box(self.sim, stone_size, stone_size, stone_height, box_asset_options)

        center_x = (num_rows - 1) * spacing / 2
        center_y = (num_cols - 1) * spacing / 2

        safe_radius = 0.5

        # Extract float values from origin once
        origin_x = origin[0].cpu().item()
        origin_y = origin[1].cpu().item()
        origin_z = origin[2].cpu().item()

        for i in range(num_rows):
            for j in range(num_cols):
                x = origin_x + i * spacing - center_x
                y = origin_y + j * spacing - center_y
                z = origin_z + stone_height / 2.0

                if np.linalg.norm([x - origin_x, y - origin_y]) < safe_radius:
                    continue

                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(x, y, z)
                self.gym.create_actor(terrain_env, box_asset, pose, f"stone_{i}_{j}", 0, 0)


    def _spawn_discrete_obstacles(self, origin, terrain_env, num_rects=10, area=40.0, min_size=0.2, max_size=0.6, max_height=0.08):
        safe_radius = 0.5

        origin_x = origin[0].cpu().item()
        origin_y = origin[1].cpu().item()
        origin_z = origin[2].cpu().item()

        for i in range(num_rects):
            size_x = np.random.uniform(min_size, max_size)
            size_y = np.random.uniform(min_size, max_size)
            size_z = np.random.uniform(0.01, max_height)

            x = origin_x + np.random.uniform(-area / 2, area / 2)
            y = origin_y + np.random.uniform(-area / 2, area / 2)
            z = origin_z + size_z / 2

            if np.linalg.norm([x - origin_x, y - origin_y]) < safe_radius:
                continue

            asset = self.create_box_asset(size_x, size_y, size_z)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(x, y, z)
            self.gym.create_actor(terrain_env, asset, pose, f"obstacle_{i}", 0, 0)




    def _spawn_sphere_obstacles(self, origin, terrain_env, num_spheres=10, area=5.0, min_radius=0.1, max_radius=0.3):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True

        safe_radius = 0.5

        # Convert torch tensor to float
        origin_x = origin[0].cpu().item()
        origin_y = origin[1].cpu().item()
        origin_z = origin[2].cpu().item()

        for i in range(num_spheres):
            radius = np.random.uniform(min_radius, max_radius)
            x = origin_x + np.random.uniform(-area / 2, area / 2)
            y = origin_y + np.random.uniform(-area / 2, area / 2)
            z = origin_z + radius * 0.5

            if np.linalg.norm([x - origin_x, y - origin_y]) < safe_radius:
                continue

            sphere_asset = self.gym.create_sphere(self.sim, radius, asset_options)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(x, y, z)
            self.gym.create_actor(terrain_env, sphere_asset, pose, f"sphere_{i}", 0, 0)


    def _spawn_stairs_4dir(self, origin, terrain_env, num_steps=10, step_height=0.05, step_length=0.4, step_width=2.0, center_size=1.0):
        # Create central flat platform
        center_thickness = 0.02
        center_asset = self.create_box_asset(center_size, center_size, center_thickness)
        center_pose = gymapi.Transform()
        center_pose.p = gymapi.Vec3(origin[0], origin[1], origin[2] + center_thickness / 2)
        self.gym.create_actor(terrain_env, center_asset, center_pose, "stairs_center", 0, 0)

        stair_asset = self.create_box_asset(step_length, step_width, step_height)

        # FORWARD stairs (+X)
        for i in range(num_steps):
            x = origin[0] + center_size / 2 + i * step_length + step_length / 2
            y = origin[1]
            z = origin[2] + (i + 0.5) * step_height
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(x, y, z)
            self.gym.create_actor(terrain_env, stair_asset, pose, f"stair_forward_{i}", 0, 0)

        # BACKWARD stairs (-X)
        for i in range(num_steps):
            x = origin[0] - center_size / 2 - i * step_length - step_length / 2
            y = origin[1]
            z = origin[2] + (i + 0.5) * step_height
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(x, y, z)
            self.gym.create_actor(terrain_env, stair_asset, pose, f"stair_backward_{i}", 0, 0)

        # LEFT stairs (+Y)
        stair_asset_y = self.create_box_asset(step_width, step_length, step_height)
        for i in range(num_steps):
            x = origin[0]
            y = origin[1] + center_size / 2 + i * step_length + step_length / 2
            z = origin[2] + (i + 0.5) * step_height
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(x, y, z)
            self.gym.create_actor(terrain_env, stair_asset_y, pose, f"stair_left_{i}", 0, 0)

        # RIGHT stairs (-Y)
        for i in range(num_steps):
            x = origin[0]
            y = origin[1] - center_size / 2 - i * step_length - step_length / 2
            z = origin[2] + (i + 0.5) * step_height
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(x, y, z)
            self.gym.create_actor(terrain_env, stair_asset_y, pose, f"stair_right_{i}", 0, 0)


    def _spawn_slope(self, origin, terrain_env, slope=0.1, size=1.0, slope_length=15.0, slope_width=15.0):
        """
        Create a small flat platform with four large slopes around it.
        """
        import math
        tilt_angle = math.atan(slope)
        sinθ = math.sin(tilt_angle)
        cosθ = math.cos(tilt_angle)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True

        # central 1x1 platform
        center_thickness = 0.02
        center_asset = self.gym.create_box(self.sim, size, size, center_thickness, asset_options)
        center_pose = gymapi.Transform()
        center_pose.p = gymapi.Vec3(origin[0], origin[1], origin[2] + center_thickness/2)
        self.gym.create_actor(terrain_env, center_asset, center_pose, "flat_center", 0, 0)

        thickness = center_thickness
        Hc = origin[2] + center_thickness
        L  = slope_length

        z_center_fb = Hc + 0.5*thickness*cosθ + 0.5*L*sinθ
        z_center_lr = Hc - 0.5*thickness*cosθ + 0.5*L*sinθ

        # slope assets with slope_width
        slope_asset_x = self.gym.create_box(self.sim, slope_length, slope_width, thickness, asset_options)
        slope_asset_y = self.gym.create_box(self.sim, slope_width, slope_length, thickness, asset_options)

        # forward slope (+X)
        forward_pose = gymapi.Transform()
        forward_pose.p = gymapi.Vec3(
            origin[0] + size/2 + L/2,
            origin[1],
            z_center_fb
        )
        forward_pose.r = gymapi.Quat.from_euler_zyx(0.0, -tilt_angle, 0.0)
        self.gym.create_actor(terrain_env, slope_asset_x, forward_pose, "slope_forward", 0, 0)

        # backward slope (-X)
        backward_pose = gymapi.Transform()
        backward_pose.p = gymapi.Vec3(
            origin[0] - size/2 - L/2,
            origin[1],
            z_center_fb
        )
        backward_pose.r = gymapi.Quat.from_euler_zyx(0.0, tilt_angle, 0.0)
        self.gym.create_actor(terrain_env, slope_asset_x, backward_pose, "slope_backward", 0, 0)

        # left slope (+Y)
        left_pose = gymapi.Transform()
        left_pose.p = gymapi.Vec3(
            origin[0],
            origin[1] + size/2 + L/2,
            z_center_lr
        )
        left_pose.r = gymapi.Quat.from_euler_zyx(tilt_angle, 0.0, 0.0)
        self.gym.create_actor(terrain_env, slope_asset_y, left_pose, "slope_left", 0, 0)

        # right slope (-Y)
        right_pose = gymapi.Transform()
        right_pose.p = gymapi.Vec3(
            origin[0],
            origin[1] - size/2 - L/2,
            z_center_lr
        )
        right_pose.r = gymapi.Quat.from_euler_zyx(-tilt_angle, 0.0, 0.0)
        self.gym.create_actor(terrain_env, slope_asset_y, right_pose, "slope_right", 0, 0)

   






    




    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        if self.cfg.control.variable_stiffness:
            base_actions = self.actions[:, :10]  # pure torques
            stiffness_damping_params = self.actions[:, 10:]

            if not self.cfg.control.fixed_stiffness:
                stiffness = stiffness_damping_params[:, :10]
                damping = stiffness_damping_params[:, 10:]
                max_stiffness = self.cfg.control.max_stiffness
                max_damping = self.cfg.control.max_damping
                self.curr_stiffness = torch.clamp((stiffness + 1.0) / 2.0 * max_stiffness, 0.0, max_stiffness)
                self.curr_damping = torch.clamp((damping + 1.0) / 2.0 * max_damping, 0.0, max_damping)

                assert not torch.isnan(self.curr_stiffness).any(), "NaN in stiffness!"
                assert not torch.isnan(self.curr_damping).any(), "NaN in damping!"

        else:
            base_actions = self.actions


        self.torques = self._compute_torques(self.actions).view(self.torques.shape)

        if not hasattr(self, 'stiffness_log'):
            self.stiffness_log = []

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques[:, :10].contiguous()))
            self.gym.simulate(self.sim)
            torch.cuda.synchronize()
            elapsed_time = self.gym.get_elapsed_time(self.sim)
            sim_time = self.gym.get_sim_time(self.sim)

            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        self.extras["observations"] = {}
        if self.privileged_obs_buf is not None:
            self.extras["observations"]["critic"] = self.privileged_obs_buf

        if hasattr(self, 'progress_buf'):
        # if hasattr(self, 'progress_buf') and self.progress_buf[0] % 50 == 0:
            print("💡 [DEBUG] Step:", self.progress_buf[0].item())
            print("  ⛓️  Gravity:", self.sim_params.gravity)
            print("  🌍 Terrain height range:", self.terrain.height_field_raw.min(), self.terrain.height_field_raw.max())
            print("  🦶 Contact force norm mean:", self.contact_forces.norm(dim=-1).mean().item())
            print("  🔧 Max torque applied:", self.torques.abs().max().item())
            print("  🧠 Action mean (first env):", self.actions[0].cpu().numpy())

        # print("✅ step() obs shape:", self.obs_buf.shape)
        # print("✅ step() action shape:", self.actions.shape)


        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras


    


    def post_physics_step(self):
        """ 
        Checks terminations, computes observations, rewards, and logs training data.
        Calls self._post_physics_step_callback() for additional physics computations.
        """
        # 🔹 Refresh physics tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # 🔹 Update step counters
        self.episode_length_buf += 1
        self.common_step_counter += 1  

        # 🔹 Select only the robot state (exclude debug cube, etc.)
        robot_root = self.root_states[self.robot_indices]  # shape: [1, 13]

        # 🔹 Compute base state (position, velocity, etc.)
        self.base_pos[:] = robot_root[:, 0:3]
        self.base_quat[:] = robot_root[:, 3:7]

        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, robot_root[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, robot_root[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 🔹 Custom physics computations (if any)
        self._post_physics_step_callback()

        # 🔹 Compute rewards, terminations, and reset conditions
        self.check_termination()
        self.compute_reward()
        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # self.reset_idx(env_ids)
        pass
        
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        # 🔹 Compute new observations
        self.compute_observations()

        # 🔹 Store last state information
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # Max joint position
        self.max_joint_pos = torch.max(self.max_joint_pos, torch.abs(self.dof_pos))

        # Joint acceleration
        joint_acc = (self.dof_vel - self.prev_joint_vel) / self.dt
        self.max_joint_acc = torch.max(self.max_joint_acc, torch.abs(joint_acc))
        self.prev_joint_vel = self.dof_vel.clone()

        # Contact force (vertical) for left/right feet
        left_z = self.contact_forces[:, self.left_foot_indices, 2].max(dim=1).values
        right_z = self.contact_forces[:, self.right_foot_indices, 2].max(dim=1).values
        self.max_contact_force_z_left = torch.max(self.max_contact_force_z_left, left_z)
        self.max_contact_force_z_right = torch.max(self.max_contact_force_z_right, right_z)


        ### 🔹 Log Training Data ###
        step = self.common_step_counter
        if hasattr(self, "log_data"):  # ✅ Ensure log_data exists
            for joint_id in range(self.num_dof):
                try:
                    self.log_data["step"].append(step)
                    self.log_data["episode"].append(self.episode_count)
                    self.log_data["joint_id"].append(joint_id)
                    self.log_data["torque"].append(self.torques[:, joint_id].mean().item())  # Mean torque across envs
                    self.log_data["action"].append(self.actions[:, joint_id].mean().item())  # Mean action across envs
                    self.log_data["reward"].append(self.rew_buf.mean().item())  # Mean reward across envs
                except Exception as e:
                    print(f"⚠️ Logging error at joint {joint_id}: {e}")

        if self.common_step_counter % 1000 == 0 and hasattr(self, "curr_stiffness") and hasattr(self, "curr_damping"):
            stiffness_mean = self.curr_stiffness.mean(dim=0).cpu().numpy()
            damping_mean = self.curr_damping.mean(dim=0).cpu().numpy()
            stiffness_min = self.curr_stiffness.min(dim=0).values.cpu().numpy()
            stiffness_max = self.curr_stiffness.max(dim=0).values.cpu().numpy()

            print(f"🔧 Step {self.common_step_counter}:")
            for i in range(5):
                print(f"  Joint {i}:  k=({stiffness_min[i]:.2f}-{stiffness_max[i]:.2f}), mean={stiffness_mean[i]:.2f} | "
                    f"d={damping_mean[i]:.2f}")
                                
        # ✅ Always log stiffness/damping, regardless of variable or fixed
        if hasattr(self, "curr_stiffness") and hasattr(self, "curr_damping"):
            self.stiffness_log.append(self.curr_stiffness.mean().item())
            self.damping_log.append(self.curr_damping.mean().item())

            # Optional: Keep logs short
            MAX_LOG_STEPS = 1000
            if len(self.stiffness_log) > MAX_LOG_STEPS:
                self.stiffness_log.pop(0)
                self.damping_log.pop(0)

            if self.common_step_counter % 1000 == 0:
                os.makedirs("logs", exist_ok=True)
                torch.save({
                    'stiffness': self.stiffness_log,  # list of floats
                    'damping': self.damping_log
                }, f"logs/stiffness_damping_step{self.common_step_counter}.pt")

                print(f"💾 Saved stiffness/damping summary at step {self.common_step_counter}")
                self.stiffness_log.clear()
                self.damping_log.clear()

        # if hasattr(self, "curr_stiffness") and hasattr(self, "prev_stiffness"):
        #     print(f"[Step {self.common_step_counter}]")
        #     print("🟦 curr_stiffness (env 0):", self.curr_stiffness[0].cpu().numpy())
        #     print("⬜ prev_stiffness (env 0):", self.prev_stiffness[0].cpu().numpy())
        #     print("📉 diff (env 0):", (self.curr_stiffness[0] - self.prev_stiffness[0]).cpu().numpy())

         # ✅ Store previous stiffness for next step
        if hasattr(self, "curr_stiffness"):
            if not hasattr(self, "prev_stiffness"):
                self.prev_stiffness = torch.zeros_like(self.curr_stiffness)
            else:
                self.prev_stiffness.copy_(self.curr_stiffness.detach())

        # print("Stiffness actions:", self.curr_stiffness[0].cpu().numpy())


        
   





    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if self.num_dof == 0:
            return torch.zeros((self.num_envs, 0), device=self.device)


        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        # print(f"🎮 Raw Actions: {actions.cpu().numpy()}")
        # print(f"📏 Scaled Actions (Torques): {actions_scaled.cpu().numpy()}")

        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
            ankle_indices = [self.dof_names.index(name) for name in self.dof_names if "Ankle" in name]
            ankle_limit = 40.0
            torques[:, ankle_indices] = torch.clamp(torques[:, ankle_indices], -ankle_limit, ankle_limit)
            # print(f"✅I AM IN TORQUE CONTROL")
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        

        if self.cfg.control.variable_stiffness:
            if self.cfg.control.stiffness_model == "parallel":
                torque_stiff = torch.zeros_like(torques)
                torque_damp = torch.zeros_like(torques)
                default_dof_pos = self.default_dof_pos.expand(self.num_envs, -1) 
                torque_stiff[:, :10] = self.curr_stiffness * (default_dof_pos[:, :10] - self.dof_pos[:, :10])
                torque_damp[:, :10] = -self.curr_damping * self.dof_vel[:, :10]
                torques = torques + torque_stiff + torque_damp

            elif self.cfg.control.stiffness_model == "series":
                default_dof_pos = self.default_dof_pos.unsqueeze(0).expand(self.num_envs, -1)
                torque_actions = actions[:, :10]
                position_scale = 0.2  # radians
                q_m = position_scale * torque_actions + default_dof_pos[:, :10]
                q_j = self.dof_pos[:, :10]
                q_dot_j = self.dof_vel[:, :10]
                k = self.curr_stiffness
                d = self.curr_damping
                torques[:, :10] = k * (q_m - q_j) - d * q_dot_j


        if self.cfg.control.fixed_stiffness:
            assert hasattr(self.cfg.control, 'fixed_stiffness_model'), "Missing fixed_stiffness_model in config."
            
            if self.cfg.control.fixed_stiffness_model == "parallel":
                default_dof_pos = self.default_dof_pos.expand(self.num_envs, -1)
                torque_stiff = self.curr_stiffness * (default_dof_pos[:, :10] - self.dof_pos[:, :10])
                torque_damp  = -self.curr_damping * self.dof_vel[:, :10]
                torques[:, :10] += torque_stiff + torque_damp

            elif self.cfg.control.fixed_stiffness_model == "series":
                default_dof_pos = self.default_dof_pos.unsqueeze(0).expand(self.num_envs, -1)
                torque_actions = actions[:, :10]
                position_modulation_scale = 0.2  # Small modulation factor (rad)
                q_m = position_modulation_scale * torque_actions + default_dof_pos[:, :10]
                q_j = self.dof_pos[:, :10]
                q_dot_j = self.dof_vel[:, :10]
                k = self.curr_stiffness  # Fixed stiffness
                d = self.curr_damping    # Fixed damping
                torques[:, :10] = k * (q_m - q_j) - d * q_dot_j 



        # print(f"⚙️ Computed Torques (Before Clipping) - Min: {torques.min().item()}, Max: {torques.max().item()}")
        torques[:, :10] = torch.clip(torques[:, :10], -self.torque_limits_tensor, self.torque_limits_tensor)
        torques_clipped = torques  # no need to clone, you just clipped in-place

        # print(f"✂️ Clipped Torques (Final Applied) - Min: {torques_clipped.min().item()}, Max: {torques_clipped.max().item()}")

        # knee_indices = [self.dof_names.index("left_knee_joint"), self.dof_names.index("right_knee_joint")]
        # torques_clipped[:, knee_indices] = torch.clamp(torques_clipped[:, knee_indices], -150.0, 80.0)


        return torques_clipped

    

    
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # reset robot DOF states
        self._reset_dofs(env_ids)

        # ✅ Sync before potentially crashing call
        torch.cuda.synchronize()

        self._reset_root_states(env_ids)

        # reset command inputs
        self._resample_commands(env_ids)

        # ✅ Safely clone all tensors that will be modified
        self.actions = self.actions.detach().clone()
        self.last_actions = self.last_actions.detach().clone()
        self.last_dof_vel = self.last_dof_vel.detach().clone()
        self.feet_air_time = self.feet_air_time.detach().clone()
        self.episode_length_buf = self.episode_length_buf.detach().clone()
        self.reset_buf = self.reset_buf.detach().clone()

        # ✅ Now it's safe to modify
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_count += 1

        
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew


    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec






    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        # self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        all_bodies = self.rigid_body_states.view(self.num_envs, -1, 13)
        print(f"all_bodies.shape = {all_bodies.shape}")
        self.rigid_body_states_view = all_bodies[:, -self.num_bodies:, :]

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

        body_names = self.gym.get_asset_rigid_body_names(self.asset)
        body_names = [str(name) for name in body_names]
        self.left_foot_indices = torch.tensor(
            [i for i, name in enumerate(body_names) if 'left_ankle' in name],
            dtype=torch.long, device=self.device
        )
        self.right_foot_indices = torch.tensor(
            [i for i, name in enumerate(body_names) if 'right_ankle' in name],
            dtype=torch.long, device=self.device
        )

        
    def _init_buffers(self):
        if not hasattr(self, "asset"):
            print("🚨 self.asset not yet set in _init_buffers()!")
            return

        print(f"DEBUG: asset = {self.asset}")

        self.num_dof = self.gym.get_asset_dof_count(self.asset)
        if self.num_dof == 0:
            print("🧪 DEBUG: Asset has 0 DOFs — skipping buffer init")

            self.dof_names = []
            self.dof_pos = torch.zeros((self.num_envs, 0), device=self.device)
            self.dof_vel = torch.zeros((self.num_envs, 0), device=self.device)
            self.dof_state = torch.zeros((self.num_envs, 0, 2), device=self.device)
            self.last_dof_pos = torch.zeros((self.num_envs, 0), device=self.device)
            self.default_dof_pos = torch.zeros(0, device=self.device)
            self.curr_stiffness = torch.zeros((self.num_envs, 0), device=self.device)
            self.curr_damping = torch.zeros((self.num_envs, 0), device=self.device)
            self.torque_limits_tensor = torch.zeros(0, device=self.device)
            self.root_states = torch.zeros((self.num_envs, 13), device=self.device)
            self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
            self.last_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
            self.commands = torch.zeros((self.num_envs, 3), device=self.device)  # [lin_vel_x, lin_vel_y, ang_vel_yaw]
            self.last_dof_vel = torch.zeros((self.num_envs, 0), device=self.device)

            # ✅ Make sure self.feet_num is defined earlier (in _create_envs)
            self.feet_air_time = torch.zeros((self.num_envs, self.feet_num), device=self.device)
            self.contact_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
            self.feet_contact_forces = torch.zeros((self.num_envs, self.feet_num), device=self.device)
            self.feet_velocities = torch.zeros((self.num_envs, self.feet_num, 3), device=self.device)
            self.feet_pos = torch.zeros((self.num_envs, self.feet_num, 3), device=self.device)
            self.feet_z = torch.zeros((self.num_envs, self.feet_num), device=self.device)
            self.command_smoothing = torch.zeros((self.num_envs, 3), device=self.device)
            self.torques = torch.zeros((self.num_envs, self.num_dof), device=self.device)
            self.base_quat = torch.zeros((self.num_envs, 4), device=self.device)  # Orientation (quaternion)
            self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
            self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
            self.base_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.rpy = torch.zeros((self.num_envs, 3), device=self.device)
            self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
            self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device)
            self.rigid_body_states = torch.zeros((self.num_envs * self.num_bodies, 13), device=self.device)
            self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
            self.max_swing_height = torch.zeros((self.num_envs, self.feet_num), device=self.device)

            return


        # ✅ Normal initialization
        super()._init_buffers()

        self.dof_state = self.dof_state.view(self.num_envs, self.num_dof, 2)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)

        self.curr_stiffness = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.curr_damping = torch.zeros((self.num_envs, self.num_dof), device=self.device)

        if self.cfg.control.fixed_stiffness:
            self.curr_stiffness[:] = torch.tensor(
                self.cfg.control.fixed_stiffness_values['stiffness'], device=self.device
            )
            self.curr_damping[:] = torch.tensor(
                self.cfg.control.fixed_stiffness_values['damping'], device=self.device
            )

        self._init_foot()
        self.last_left_contact = torch.zeros(self.num_envs, device=self.device)
        self.last_right_contact = torch.zeros(self.num_envs, device=self.device)
        self.max_swing_height = torch.zeros((self.num_envs, self.feet_num), device=self.device)

        self.torque_limits_tensor = torch.zeros(self.num_dof, device=self.device)
        for i, dof_name in enumerate(self.dof_names):
            name = dof_name.lower()
            if 'hip_yaw' in name:
                self.torque_limits_tensor[i] = self.cfg.control.torque_limits['hip_yaw']
            elif 'hip_roll' in name:
                self.torque_limits_tensor[i] = self.cfg.control.torque_limits['hip_roll']
            elif 'hip_pitch' in name:
                self.torque_limits_tensor[i] = self.cfg.control.torque_limits['hip_pitch']
            elif 'knee' in name:
                self.torque_limits_tensor[i] = self.cfg.control.torque_limits['knee']
            elif 'ankle' in name:
                self.torque_limits_tensor[i] = self.cfg.control.torque_limits['ankle']
            elif 'torso' in name:
                self.torque_limits_tensor[i] = self.cfg.control.torque_limits['torso']
            elif 'shoulder' in name:
                self.torque_limits_tensor[i] = self.cfg.control.torque_limits['shoulder']
            elif 'elbow' in name:
                self.torque_limits_tensor[i] = self.cfg.control.torque_limits['elbow']
            else:
                self.torque_limits_tensor[i] = 1000.0  # fallback

        self.default_dof_pos = torch.zeros(self.num_dof, device=self.device)
        for i, name in enumerate(self.dof_names):
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles.get(name, 0.0)





    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        swing_mask = ~contact
        foot_heights = self.feet_pos[:, :, 2] - 0.07  # ankle to sole
        self.max_swing_height = torch.maximum(self.max_swing_height, foot_heights * swing_mask.float())

        return super()._post_physics_step_callback()
    
    def _compute_zmp(self):
        """
        Compute the Zero Moment Point (ZMP) using foot contact forces.
        Returns ZMP_x and ZMP_y for stability control.
        """
        if hasattr(self, "contact_forces"):
            contact_forces = self.contact_forces  # Shape: (num_envs, num_bodies, 3)
            foot_positions = self.feet_pos if hasattr(self, "feet_pos") else torch.zeros_like(contact_forces)
            if hasattr(self, "feet_indices"):
                foot_indices = self.feet_indices.flatten()  # Ensure it's 1D
            else:
                foot_indices = torch.tensor([0, 1], device=self.device)  # Default (replace with actual foot indices)
            Fz = contact_forces[:, foot_indices, 2]  # Shape: (num_envs, num_feet)
            foot_positions = foot_positions[:, :len(foot_indices), :]  # Ensure correct shape
            valid_feet_mask = Fz > 1e-3  # Consider only feet with non-negligible force
            valid_Fz = Fz * valid_feet_mask  # Zero out forces for non-contacting feet
            # Count active feet
            active_feet_count = torch.sum(valid_feet_mask, dim=1, keepdim=True)
            # Case 1: If only **one foot** is in contact, set ZMP at that foot's position
            single_foot_mask = (active_feet_count == 1).squeeze(-1)
            multi_foot_mask = (active_feet_count > 1).squeeze(-1)
            # ZMP for **single-foot contact** → Directly set to foot position
            zmp_x = torch.zeros_like(Fz[:, 0])  # Init
            zmp_y = torch.zeros_like(Fz[:, 0])  # Init
            zmp_x[single_foot_mask] = foot_positions[single_foot_mask, valid_feet_mask[single_foot_mask].nonzero()[:, 1], 0]
            zmp_y[single_foot_mask] = foot_positions[single_foot_mask, valid_feet_mask[single_foot_mask].nonzero()[:, 1], 1]
            # ZMP for **multiple feet in contact** → Compute weighted average
            if torch.any(multi_foot_mask):
                force_ratio = valid_Fz[multi_foot_mask] / (torch.sum(valid_Fz[multi_foot_mask], dim=1, keepdim=True) + 1e-6)
                force_weight = torch.exp(-5 * (1 - force_ratio))  # Exponential smoothing
                zmp_x[multi_foot_mask] = torch.sum(force_weight * valid_Fz[multi_foot_mask] * foot_positions[multi_foot_mask, :, 0], dim=1) / torch.sum(force_weight * valid_Fz[multi_foot_mask], dim=1)
                zmp_y[multi_foot_mask] = torch.sum(force_weight * valid_Fz[multi_foot_mask] * foot_positions[multi_foot_mask, :, 1], dim=1) / torch.sum(force_weight * valid_Fz[multi_foot_mask], dim=1)
            # Clamping to the foot positions
            zmp_x = torch.clamp(zmp_x, torch.min(foot_positions[:, :, 0]), torch.max(foot_positions[:, :, 0]))
            zmp_y = torch.clamp(zmp_y, torch.min(foot_positions[:, :, 1]), torch.max(foot_positions[:, :, 1]))
            # Compute support polygon
            support_polygon_x = [torch.min(foot_positions[:, :, 0], dim=1)[0], torch.max(foot_positions[:, :, 0], dim=1)[0]]
            support_polygon_y = [torch.min(foot_positions[:, :, 1], dim=1)[0], torch.max(foot_positions[:, :, 1], dim=1)[0]]

            # Debugging output
            # print(f"👣 Active Feet Count: {active_feet_count.squeeze(-1)}")
            # for i in range(Fz.shape[1]):  # Loop over feet
            #     print(f"🦶 Foot {i}: Force = {Fz[:, i].mean().item():.3f}")
            # print(f"📏 Foot Positions (X): min {foot_positions[:, :, 0].min().item()}, max {foot_positions[:, :, 0].max().item()}")
            # print(f"📏 Foot Positions (Y): min {foot_positions[:, :, 1].min().item()}, max {foot_positions[:, :, 1].max().item()}")
            # print(f"🟢 ZMP_x: {zmp_x.mean().item()}, ZMP_y: {zmp_y.mean().item()}")
            # print("👣 Foot Indices:", foot_indices)
            # print("👣 Foot Positions:", foot_positions[:, :, :2])  # X and Y

            return zmp_x, zmp_y, support_polygon_x, support_polygon_y
    
        return None, None, None, None
    

    def _load_masses(self):
        """ Get rigid body masses from Isaac Gym and store them in self.masses. """
        masses_list = []

        for i in range(self.num_envs):
            actor_props = self.gym.get_actor_rigid_body_properties(self.envs[i], self.actor_handles[i])
            masses = torch.tensor([prop.mass for prop in actor_props], device=self.device)
            masses_list.append(masses)

        self.masses = torch.stack(masses_list)  # Shape: (num_envs, num_bodies)
        # print(f"✅ Loaded masses: {self.masses.shape}")

   

    def _compute_com(self):
        """ Compute Center of Mass (CoM) using rigid body positions and masses. """
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if not hasattr(self, "rigid_body_states"):
            print("🚨 Missing `rigid_body_states` in environment.")
            return None, None, None
        if not hasattr(self, "masses") or self.masses is None:
            print("🚨 `self.masses` is missing! Calling `_load_masses()` to initialize it.")
            self._load_masses()  # Call _load_masses() before proceeding
        num_envs = self.num_envs  # Number of environments
        if hasattr(self, "masses"):
            num_bodies = self.masses.shape[1]  # Ensure we match the number of bodies
        else:
            print("🚨 `self.masses` is still missing after calling `_load_masses()`. Using default value.")
            num_bodies = 11  # Fallback value (adjust as needed)
        self.rigid_body_states = self.rigid_body_states.view(num_envs, -1, 13)
        body_pos = self.rigid_body_states[:, :num_bodies, :3]  # Extract (x, y, z) positions
        if not hasattr(self, "masses") or self.masses is None:
            print("🚨 Masses attribute missing! Using uniform mass assumption.")
            masses = torch.ones((num_envs, num_bodies, 1), device=self.device)  # Assume uniform mass
        else:
            masses = self.masses.unsqueeze(-1)  # Ensure shape (num_envs, num_bodies, 1)
        total_mass = torch.sum(masses, dim=1, keepdim=True) + 1e-6  # Avoid division by zero
        com = torch.sum(body_pos * masses, dim=1, keepdim=False) / total_mass.squeeze(1)

        return com[:, 0], com[:, 1], com[:, 2]

 
    
    def compute_observations(self):
        """ Computes observations """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)

        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase
        ), dim=-1)

        self.privileged_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase
        ), dim=-1)

        if self.cfg.control.variable_stiffness:
            max_stiffness = self.cfg.control.max_stiffness
            max_damping = self.cfg.control.max_damping
            stiffness_damping_obs = torch.cat([
                self.curr_stiffness / max_stiffness,
                self.curr_damping / max_damping
            ], dim=-1)

            # Add to both obs and privileged_obs
            self.obs_buf = torch.cat((self.obs_buf, stiffness_damping_obs), dim=-1)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, stiffness_damping_obs), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def get_observations(self):

        # print("✅ obs_buf shape:", self.obs_buf.shape)
        # print("✅ privileged_obs_buf shape:", self.privileged_obs_buf.shape)

        return self.obs_buf, {
            "observations": {
                "obs": self.obs_buf,
                "critic": self.privileged_obs_buf
            }
        }




    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def save_logs(self):
        """ Save logs to CSV after every step to avoid missing data """
        df = pd.DataFrame(self.log_data)
        if len(df) == 0:
            print("⚠️ No data to log!")  # Debug print
            return  # No data to save
        
        log_dir = "legged_gym/logs"
        os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists
        log_file = os.path.join(log_dir, "training_logs.csv")

        if os.path.exists(log_file):
            df.to_csv(log_file, mode='a', header=False, index=False)  # Append mode
        else:
            df.to_csv(log_file, mode='w', header=True, index=False)  # Create new file

    
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > self.cfg.rewards.contact_force_threshold
            match = torch.logical_not(torch.logical_xor(contact, is_stance)) # Foot is supposed to be on the ground and is on the ground OR supposed to be in the air and is in the air 
            res += match.float()
        return res / self.feet_num  # optional normalization
    

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0 # True: foot in contact
        sole_height = self.feet_pos[:, :, 2] - 0.07  # ankle to sole correction
        swing_feet = ~contact
        target_clearance = 0.05  # meters
        low_clearance = torch.clamp(target_clearance - sole_height, min=0.0)
        reward = torch.square(low_clearance) * swing_feet
        return torch.sum(reward, dim=1)
    
    # def _reward_feet_swing_height(self):
    #     contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
    #     pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
    #     return torch.sum(pos_error, dim=(1))

    
    def _reward_alive(self):
        # Reward for staying alive
        # return 1.0
        return torch.ones(self.num_envs, device=self.device)
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1. # Checks if each foot is in contact 
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1) # For feet not in contact → velocity becomes zero, For feet in contact → keep their velocity.
        penalize = torch.square(contact_feet_vel[:, :, :3]) # Takes the squared velocity for each contacting foot.
        return torch.sum(penalize, dim=(1,2)) # penalizing contact with nonzero velocity — i.e., feet that are sliding while in contact.
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[0,1,5,6]]), dim=1)
    
    def _reward_zmp_error(self):
        """
        Penalizes large deviations of the Zero Moment Point (ZMP) from the support polygon.
        The penalty increases when the ZMP moves outside the support boundaries.
        """
        zmp_x, zmp_y, support_x, support_y = self._compute_zmp()
        if zmp_x is None or zmp_y is None:
            return torch.tensor(0.0, device=self.device)
        zmp_x_min = support_x[0]  # Min X boundary
        zmp_x_max = support_x[1]  # Max X boundary
        zmp_y_min = support_y[0]  # Min Y boundary
        zmp_y_max = support_y[1]  # Max Y boundary

        zmp_x_error = torch.clamp(zmp_x - zmp_x_max, min=0) + torch.clamp(zmp_x_min - zmp_x, min=0)
        zmp_y_error = torch.clamp(zmp_y - zmp_y_max, min=0) + torch.clamp(zmp_y_min - zmp_y, min=0)

        # Compute total ZMP penalty (higher if ZMP is far from the center)
        zmp_penalty = torch.square(zmp_x_error) + torch.square(zmp_y_error)
        return zmp_penalty   # Scale appropriately
    
    def _reward_step_alternation(self):
        left_idx = self.feet_indices[0]
        right_idx = self.feet_indices[1]
        left_contact = self.contact_forces[:, left_idx, 2] > self.cfg.rewards.contact_force_threshold # check if the left foot are currently in contact with the ground
        right_contact = self.contact_forces[:, right_idx, 2] > self.cfg.rewards.contact_force_threshold # check if the right foot are currently in contact with the ground
        alternating = (left_contact.int() - right_contact.int()).abs()
        return alternating.float()

    def _reward_ankle_torque_penalty(self):
        ankle_indices = [4,9]  
        ankle_torques = self.torques[:, :10][:, ankle_indices]
        torque_norm = torch.sum(torch.square(ankle_torques), dim=1) 
        return torque_norm 


    def _reward_contact_symmetry(self):
        left = torch.sum(self.contact_forces[:, self.left_foot_indices, 2], dim=1)
        right = torch.sum(self.contact_forces[:, self.right_foot_indices, 2], dim=1)
        diff = left - right
        return -torch.square(diff)
    
    def _reward_tracking_lin_vel_y(self):
        # Tracking of sideways (Y) linear velocity
        error_y = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-error_y / self.cfg.rewards.tracking_sigma)
    
    def _reward_anti_hop(self):
        left_idx = self.feet_indices[0]
        right_idx = self.feet_indices[1]
        left_contact = self.contact_forces[:, left_idx, 2] > self.cfg.rewards.contact_force_threshold
        right_contact = self.contact_forces[:, right_idx, 2] > self.cfg.rewards.contact_force_threshold
        both_feet_up = (~left_contact & ~right_contact)
        penalty = both_feet_up.float()
        return penalty
    
  
    def _reward_smooth_landing(self):
        contact = self.contact_forces[:, self.feet_indices, 2]
        landing_force = torch.max(contact, dim=1).values
        return landing_force
    
    def _reward_foot_impact_velocity(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0  # [num_envs, num_feet]
        vertical_vel = self.feet_vel[:, :, 2]  # [num_envs, num_feet] vertical velocity of feet
        impact_vel = torch.abs(vertical_vel) * contact  # only count velocity at contact
        return torch.sum(impact_vel, dim=1)  # penalize large vertical velocity at contact  

    def _reward_feet_lift_clearance(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        swing_feet = ~contact
        foot_heights = self.feet_pos[:, :, 2] - 0.07  # ankle to sole offset
        # Between 10cm and 15cm: increasing reward
        # Past 15cm: plateau (don't punish, just stop rewarding)
        clearance_threshold = 0.02  # start of reward
        max_bonus_height = 0.07     # cap reward here
        range_width = max_bonus_height - clearance_threshold
        # Linear reward with soft cap
        clearance = torch.clamp((foot_heights - clearance_threshold) / range_width, min=0.0, max=1.0)
        # Only reward swing feet
        reward = clearance * swing_feet.float()
        return torch.sum(reward, dim=1)

    def _reward_hip_pitch_limit_penalty(self):
        # Get hip pitch joint angles (adjust indices if needed)
        left_hip_pitch = self.dof_pos[:, 2]  # L_HipPitch
        right_hip_pitch = self.dof_pos[:, 7]  # R_HipPitch
        # Limit (±90 degrees = π/2 radians)
        limit = 90 * torch.pi / 180
        # Penalize only the portion that exceeds the limit
        left_penalty = torch.clamp(torch.abs(left_hip_pitch) - limit, min=0.0)
        right_penalty = torch.clamp(torch.abs(right_hip_pitch) - limit, min=0.0)
        # Optionally square for stronger punishment on large violations
        penalty = torch.square(left_penalty) + torch.square(right_penalty)
        return penalty
    
    
    def _reward_feet_touchdown(self):
        # Encourage the shortest possible air time (i.e., fast stepping)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        reward = (1.0 / (self.feet_air_time + 1e-4)) * first_contact

        self.feet_air_time *= ~contact_filt
        return torch.sum(reward, dim=1)


    def _reward_initial_torque_suppression(self):
        ramp = (self.episode_length_buf < 20).float() * (1.0 - self.episode_length_buf / 20.0)
        return torch.sum(torch.square(self.torques[:, :10]), dim=1) * ramp
    
    def _reward_stay_upright(self):
        return (self.base_pos[:, 2] > 0.81).float()

    
    def _reward_contact_symmetry_z(self):
        left_force = torch.sum(self.contact_forces[:, self.left_foot_indices, 2], dim=1)
        right_force = torch.sum(self.contact_forces[:, self.right_foot_indices, 2], dim=1)
        force_diff = left_force - right_force
        return torch.square(force_diff)  # penalize asymmetric vertical force
    
    
    def _reward_hip_pitch_symmetry(self):
        if not hasattr(self, "last_swing_amp"):
            self.last_swing_amp = torch.zeros((self.num_envs,), device=self.device)
            self.last_swing_leg = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)  # True = Left

        left_hip = self.dof_pos[:, 2]
        right_hip = self.dof_pos[:, 7]
        left_swing = self.leg_phase[:, 0] >= 0.55
        right_swing = self.leg_phase[:, 1] >= 0.55

        reward = torch.zeros(self.num_envs, device=self.device)

        # When left leg swings
        new_left_swing = left_swing & (~self.last_swing_leg)
        amp_left = torch.abs(left_hip)
        reward += -torch.square(amp_left - self.last_swing_amp) * new_left_swing.float()

        self.last_swing_amp = torch.where(new_left_swing, amp_left, self.last_swing_amp)
        self.last_swing_leg = torch.where(new_left_swing, torch.ones_like(self.last_swing_leg), self.last_swing_leg)

        # When right leg swings
        new_right_swing = right_swing & self.last_swing_leg
        amp_right = torch.abs(right_hip)
        reward += -torch.square(amp_right - self.last_swing_amp) * new_right_swing.float()

        self.last_swing_amp = torch.where(new_right_swing, amp_right, self.last_swing_amp)
        self.last_swing_leg = torch.where(new_right_swing, torch.zeros_like(self.last_swing_leg), self.last_swing_leg)

        return reward





    def _reward_foot_forward_swing(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        swing_feet = ~contact  # boolean mask
        foot_forward = self.feet_pos[:, :, 0] - self.base_pos[:, 0].unsqueeze(1)  # relative to robot
        reward = foot_forward * swing_feet.float()
        return torch.sum(reward, dim=1)

    def _reward_peak_foot_lift(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        first_contact = contact & ~self.last_contacts  # detect touchdown
        self.last_contacts = contact  # update contact buffer
        target = 0.15  # meters 
        error = self.max_swing_height - target
        reward = torch.square(error) * first_contact.float()
        self.max_swing_height[first_contact] = 0.0

        return torch.sum(reward, dim=1) * 10000
    
    def _reward_knee_pitch_symmetry(self):
        # Index 3 = L_Knee, Index 8 = R_Knee
        left = self.dof_pos[:, 3]
        right = self.dof_pos[:, 8]
        return torch.square(left - right)

    def _reward_power_stiffness_penalty(self):
        # penalize torque magnitude if stiffness is too low (e.g., soft joints flopping)
        torque = self.torques[:, :10]
        stiffness = self.curr_stiffness / self.cfg.control.max_stiffness + 1e-4
        penalty = torch.sum(torque**2 / stiffness, dim=1)
        return penalty

    def _reward_power_damping_penalty(self):
        # penalize torque magnitude if damping is too low (e.g., unstable under impact)
        torque = self.torques[:, :10]
        damping = self.curr_damping / self.cfg.control.max_damping + 1e-4
        penalty = torch.sum(torque**2 / damping, dim=1)
        return penalty

    def _reward_phase_stiffness_profile(self):
        # For example: push-off and landing → higher stiffness
        max_stiffness = self.cfg.control.max_stiffness
        target_value = 0.6 * max_stiffness
        push_off = self.leg_phase[:, 0] < 0.2
        landing = self.leg_phase[:, 0] > 0.8
        target_stiffness = torch.zeros_like(self.curr_stiffness)
        target_stiffness += target_value * push_off.unsqueeze(1)
        target_stiffness += target_value * landing.unsqueeze(1)
        error = torch.square(self.curr_stiffness - target_stiffness)
        return -torch.mean(error, dim=1)
    
    def _reward_stiffness_variation(self):
        diff = self.curr_stiffness - self.prev_stiffness
        return torch.mean(torch.abs(diff), dim=1)
    
    def _reward_ankle_stiffness_usage(self):
        # Encourage ankle pitch joints (index 4 and 9) to be nonzero stiffness
        ankle_pitch_L = self.curr_stiffness[:, 4]
        ankle_pitch_R = self.curr_stiffness[:, 9]
        ankle_stiff_mean = (ankle_pitch_L + ankle_pitch_R) / 2.0
        return ankle_stiff_mean  # reward will be higher if stiffness > 0

    def _reward_torso_pitch(self):
        return torch.square(self.projected_gravity[:, 0])
    
    def _reward_low_swing_foot_clearance(self):
        # Parameters
        target_clearance = 0.03  # 3 cm
        alpha = 100.0
        # Get contact info as in feet_air_time
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        # Use existing feet_pos buffer
        foot_heights = self.feet_pos[:, :, 2]  # vertical height of foot (shape: [num_envs, num_feet])
        # Boolean mask for swing phase
        swing_mask = ~contact_filt
        # Compute squared error from desired height
        height_error = foot_heights - target_clearance
        penalty = -alpha * (height_error ** 2) * swing_mask
        # Sum over feet, return per-env reward
        return torch.sum(penalty, dim=1)


