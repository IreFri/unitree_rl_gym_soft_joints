import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import glob
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry
from isaacgym import gymapi
from isaacgym import gymtorch
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import subprocess
import json
import time
import cv2 
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QSlider, QLabel
from PyQt5.QtCore import Qt
import argparse


EXPORT_POLICY = True
RECORD_FRAMES = False
MOVE_CAMERA = True
JOYSTICK_GUI = False

CAMERA_VIEW = "BACK"  # Options: "BACK", "FRONT", "RIGHT", "LEFT", "TOP"

plots_dir = "legged_gym/plots"
plots_path = os.path.join(plots_dir, "robot_log.json")
video_out_path = os.path.join(plots_dir, "H1_rigid.mp4")
video_out_path_hq = os.path.join(plots_dir, "simulation_video_HQ.mp4")
max_duration = 60  # Stop after 30 seconds



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

def convert_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  
    elif isinstance(obj, list):
        return [convert_to_list(item) for item in obj] 
    else:
        return obj 
    
def convert_tensors(obj):
    """ Recursively convert all torch tensors in obj to Python-native types. """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert tensor to list (safe for JSON)
    elif isinstance(obj, dict):
        return {k: convert_tensors(v) for k, v in obj.items()}  # Recursively convert dict
    elif isinstance(obj, list):
        return [convert_tensors(i) for i in obj]  # Recursively convert list
    return obj 

def warmup(env, actor_critic, num_steps=20):
    """
    Warmup the robot: reset orientation, stop velocities, set default DOF pos/vel,
    and stabilize for a few steps before applying any motion command.
    """
    # 0. Reset robot pose and velocity manually
    env.root_states[:, 0:3] = torch.tensor([[0.0, 0.0, 1.0]], device=env.device)  # x, y, z
    env.root_states[:, 3:7] = torch.tensor([[0, 0, 0, 1]], device=env.device)     # quaternion
    env.root_states[:, 7:13] = 0.0                                                 # linear + angular velocities

    # 1. Reset DOF positions and velocities
    gym_root_states_raw = env.gym.acquire_actor_root_state_tensor(env.sim)
    gym_root_states = gymtorch.wrap_tensor(gym_root_states_raw).view(-1, 13)
    gym_root_states[0].copy_(env.root_states[0])

    gym_dof_states_raw = env.gym.acquire_dof_state_tensor(env.sim)
    gym_dof_states = gymtorch.wrap_tensor(gym_dof_states_raw).view(env.num_envs * env.num_dof, 2)
    gym_dof_states[:, 0] = env.default_dof_pos.view(-1)  # Default joint angles
    gym_dof_states[:, 1] = 0.0                           # Zero joint velocities

    # 2. Apply back to simulation
    env.gym.set_actor_root_state_tensor(env.sim, gym_root_states_raw)
    env.gym.set_dof_state_tensor(env.sim, gym_dof_states_raw)

    # 3. Set zero commands (no movement)
    env.commands[:] = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=env.device)

    # 4. Take a few warmup steps with the policy
    obs, _ = env.get_observations()
    for _ in range(num_steps):
        actions = actor_critic.act_inference(obs)
        obs, _, _, _, _ = env.step(actions.detach())

    print(f"🔥 Warmup finished: {num_steps} stabilization steps")


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
    corrected_yaw_rate = heading_error * 10.0

    return corrected_yaw_rate
    
if JOYSTICK_GUI:

    class JoystickGUI(QWidget):

        command_values = {
            "forward": torch.tensor([[1.0, 0.0, 0.0]]),  # vx, vy, yaw_rate
            "backward": torch.tensor([[-1.0, 0.0, 0.0]]), 
            "left": torch.tensor([[0.0, 1.0, 0.0]]),      
            "right": torch.tensor([[0.0, -1.0, 0.0]]),    
            "turn_left": torch.tensor([[0.0, 0.0, 1.0]]), 
            "turn_right": torch.tensor([[0.0, 0.0, -1.0]]), 
            "stop": torch.tensor([[0.0, 0.0, 0.0]])       
        }
        
        def __init__(self):
            super().__init__()
            print("✅ GUI is being created!")

            # Initialize velocity values BEFORE calling initUI()
            self.velocity_x = 0.0  # Forward/Backward velocity
            self.velocity_y = 0.0  # Lateral velocity
            self.yaw_rate = 0.0    # Turning rate
            self.current_command = torch.tensor([[0.0, 0.0, 0.0]])  # Initial stop command
            self.active_button = None
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
            self.initUI()
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            self.show()
            self.raise_()
            self.activateWindow()
            QApplication.processEvents()
            time.sleep(0.5)

        def initUI(self):
            self.setWindowTitle('Robot Controller')
            self.setGeometry(100, 100, 300, 500)

            layout = QVBoxLayout()

            # Slider for Forward/Backward Velocity (vx)
            self.vel_x_label = QLabel(f"Forward/Backward: {self.velocity_x:.2f} m/s", self)
            self.vel_x_slider = QSlider(Qt.Horizontal)
            self.vel_x_slider.setMinimum(0)   # 0 m/s
            self.vel_x_slider.setMaximum(150) # Max 1.5 m/s (scaled by 100)
            self.vel_x_slider.setValue(0)     # Default 0.0 m/s
            self.vel_x_slider.valueChanged.connect(self.update_velocity_x)

            # Slider for Lateral Velocity (vy)
            self.vel_y_label = QLabel(f"Lateral: {self.velocity_y:.2f} m/s", self)
            self.vel_y_slider = QSlider(Qt.Horizontal)
            self.vel_y_slider.setMinimum(-150) # -1.5 m/s (left)
            self.vel_y_slider.setMaximum(150)  # 1.5 m/s (right)
            self.vel_y_slider.setValue(0)      # Default 0.0 m/s
            self.vel_y_slider.valueChanged.connect(self.update_velocity_y)

            # Slider for Turning Rate (yaw)
            self.yaw_label = QLabel(f"Turning Rate: {self.yaw_rate:.2f} rad/s", self)
            self.yaw_slider = QSlider(Qt.Horizontal)
            self.yaw_slider.setMinimum(-100) # -1.0 rad/s (right turn)
            self.yaw_slider.setMaximum(100)  # 1.0 rad/s (left turn)
            self.yaw_slider.setValue(0)      # Default 0.0 rad/s
            self.yaw_slider.valueChanged.connect(self.update_yaw_rate)


            # Movement Buttons
            self.btn_forward = QPushButton('Forward', self)
            self.btn_backward = QPushButton('Backward', self)
            self.btn_left = QPushButton('Left', self)
            self.btn_right = QPushButton('Right', self)
            self.btn_turn_left = QPushButton('Turn Left', self)
            self.btn_turn_right = QPushButton('Turn Right', self)
            self.btn_stop = QPushButton('Stop', self)

            # Connect buttons to movement commands
            self.btn_forward.clicked.connect(lambda: self.set_command("forward"))
            self.btn_backward.clicked.connect(lambda: self.set_command("backward"))
            self.btn_left.clicked.connect(lambda: self.set_command("left"))
            self.btn_right.clicked.connect(lambda: self.set_command("right"))
            self.btn_turn_left.clicked.connect(lambda: self.set_command("turn_left"))
            self.btn_turn_right.clicked.connect(lambda: self.set_command("turn_right"))
            self.btn_stop.clicked.connect(lambda: self.set_command("stop"))

            # Add widgets to layout
            layout.addWidget(self.vel_x_label)
            layout.addWidget(self.vel_x_slider)
            layout.addWidget(self.vel_y_label)
            layout.addWidget(self.vel_y_slider)
            layout.addWidget(self.yaw_label)
            layout.addWidget(self.yaw_slider)

            layout.addWidget(self.btn_forward)
            layout.addWidget(self.btn_backward)
            layout.addWidget(self.btn_left)
            layout.addWidget(self.btn_right)
            layout.addWidget(self.btn_turn_left)
            layout.addWidget(self.btn_turn_right)
            layout.addWidget(self.btn_stop)

            self.setLayout(layout)

        def update_velocity_x(self, value):
            """Update forward/backward velocity when slider changes."""
            self.velocity_x = value / 100.0  # Scale slider (0 to 2.0 m/s)
            self.vel_x_label.setText(f"Forward/Backward: {self.velocity_x:.2f} m/s")

            # If the last movement was backward, keep it negative
            if self.active_button == self.btn_backward:
                self.velocity_x = -self.velocity_x  # Keep the backward direction

            # Apply the updated velocity while keeping direction
            if self.velocity_x > 0:
                self.set_command("forward")
            elif self.velocity_x < 0:
                self.set_command("backward")
            else:
                self.set_command("stop")  # Reset if velocity is zero


        def update_velocity_y(self, value):
            """Update lateral velocity when slider changes."""
            self.velocity_y = value / 100.0  # Scale slider (-2.0 to 2.0 m/s)
            self.vel_y_label.setText(f"Lateral: {self.velocity_y:.2f} m/s")

            # Apply correct direction
            if self.velocity_y > 0:
                self.set_command("right")
            elif self.velocity_y < 0:
                self.set_command("left")
            else:
                self.set_command("stop")  # Reset if velocity is zero


            # If positive, move right; if negative, move left
            if self.velocity_y > 0:
                self.set_command("right")  
            elif self.velocity_y < 0:
                self.set_command("left")
            else:
                self.set_command("stop")  # Reset if velocity is zero


        def update_yaw_rate(self, value):
            """Update turning rate when slider changes."""
            self.yaw_rate = value / 100.0  # Scale slider (-1.0 to 1.0 rad/s)
            self.yaw_label.setText(f"Turning Rate: {self.yaw_rate:.2f} rad/s")

            # Apply correct direction
            if self.yaw_rate > 0:
                self.set_command("turn_left")
            elif self.yaw_rate < 0:
                self.set_command("turn_right")
            else:
                self.set_command("stop")  # Reset if yaw rate is zero




        def set_command(self, command):
            """Set movement command while keeping updated velocity values from sliders."""
            if command == "stop":
                self.velocity_y = 0.0
                self.velocity_x = 0.0
                self.yaw_rate = 0.0
    
            base_command = torch.tensor([[self.velocity_x, -self.velocity_y, self.yaw_rate]])  # Use slider values
            self.current_command = base_command  # Update current command
            # Reset all button colors
            for button in [self.btn_forward, self.btn_backward, self.btn_left, self.btn_right, 
                        self.btn_turn_left, self.btn_turn_right, self.btn_stop]:
                button.setStyleSheet("")  # Reset color

            # Highlight the active button
            button_map = {
                "forward": self.btn_forward,
                "backward": self.btn_backward,
                "left": self.btn_left,
                "right": self.btn_right,
                "turn_left": self.btn_turn_left,
                "turn_right": self.btn_turn_right,
                "stop": self.btn_stop
            }
            if self.active_button:
                self.active_button.setStyleSheet("")  
            self.active_button = button_map[command]
            self.active_button.setStyleSheet("background-color: yellow")

            # Force GUI update
            QApplication.processEvents()


        def get_current_command(self, num_envs):
            """Return the current movement command to the environment."""
            command = self.current_command  # Get latest command
            command_padded = torch.cat((command, torch.zeros((1, 1))), dim=1)  # Ensure it has 4 dimensions
            expanded_command = command_padded.expand(num_envs, -1)  # Match shape [num_envs, 4]

            return expanded_command



def play(args):
    import isaacgym.gymtorch as gymtorch

    if JOYSTICK_GUI:
        app = QApplication(sys.argv)
        gui = JoystickGUI()
        gui.show()
        gui.raise_()
        gui.activateWindow()
        QApplication.processEvents()
        time.sleep(0.5)
    else:
        app = None
        gui = None


    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.domain_rand.push_robots = False

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.sim_device = "cuda:0" if device == "cuda" else "cpu"
    args.physics_engine = gymapi.SIM_PHYSX
    args.headless = False

    env, _ = task_registry.make_env(args.task, args, env_cfg)
    env.reset()
    # print("🔧 DOF Names (first 10):", env.dof_names[:10])
    if hasattr(env, "get_com"):
        com_pos = env.get_com()
    else:
        com_pos = env.base_pos
    obs, _ = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env, args.task, args, train_cfg)
    actor_critic = ppo_runner.alg.actor_critic 

    env.reset()
    com_pos = env.get_com() if hasattr(env, "get_com") else env.base_pos  

    # Warmup and velocity corrections
    warmup(env, actor_critic, num_steps=20)
    correct_lateral_velocity(env)


    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    
    if RECORD_FRAMES:
        video_writer = None  
        # Create the camera sensor
        frame_width = 1280   # Default width (change if needed)
        frame_height = 720  # Default height (change if needed)
        cam_props = gymapi.CameraProperties()
        cam_props.width = frame_width  # ✅ No more UnboundLocalError!
        cam_props.height = frame_height
        cam_handle = env.gym.create_camera_sensor(env.envs[0], cam_props)  

        # Attach the camera to the robot
        robot_pos = com_pos[0].tolist()  
        camera_position = gymapi.Vec3(robot_pos[0] - 2.0, robot_pos[1], robot_pos[2] + 1.0)  
        cam_transform = gymapi.Transform()
        cam_transform.p = camera_position 
        env.gym.attach_camera_to_body(
            cam_handle,
            env.envs[0],
            env.gym.get_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 0),
            gymapi.Transform(),
            gymapi.CameraFollowMode.FOLLOW_POSITION  
        )

        cam2_props = gymapi.CameraProperties()
        cam2_props.width = 3840  
        cam2_props.height = 2160  
        cam2_handle = env.gym.create_camera_sensor(env.envs[0], cam2_props)        

        # Set Camera Position (relative to robot)
        if CAMERA_VIEW == "BACK":
            fixed_cam_position = gymapi.Vec3(robot_pos[0] - 2.0, robot_pos[1], robot_pos[2] + 1)  
            roll, pitch, yaw = [0,30,0]
        elif CAMERA_VIEW == "LEFT":
            fixed_cam_position = gymapi.Vec3(robot_pos[0] + 4.0, robot_pos[1] + 8.0, robot_pos[2] + 1)  
            roll, pitch, yaw = [0,0,-90]  
        elif CAMERA_VIEW == "RIGHT":
            fixed_cam_position = gymapi.Vec3(robot_pos[0], robot_pos[1] - 3.0, robot_pos[2] + 1)  
            roll, pitch, yaw = [0,0,90] 
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


    lstm_hidden_history = []
    lstm_cell_history = []

    # ⏳ Simulation loop
    start_time = time.time()  # Start timing
    last_frame_time = time.time()
    desired_heading = 0.0
    correct_lateral = False
    step_idx = 0
    if not JOYSTICK_GUI:
        vx = 0.0
        vy = 0.0
        yaw_cmd = 0.0

    while True:
        current_time = time.time()

        if JOYSTICK_GUI:
            command = gui.get_current_command(env.num_envs)
            env.commands[:] = command

            vx = command[0, 0].item()
            vy = command[0, 1].item()
            yaw_cmd = command[0, 2].item()

            # 🧠 Decide if we want to apply lateral correction or yaw correction
            if abs(yaw_cmd) > 0.1:   # If user is trying to turn
                desired_heading = None
                correct_lateral = False
            elif abs(vy) > 0.1:      # If user is moving left/right
                desired_heading = None
                correct_lateral = True
            else:                   # Just moving forward/backward
                desired_heading = 0.0
                correct_lateral = False

            # 🔥 Apply yaw correction
            if desired_heading is not None:
                corrected_yaw_rate = adjust_yaw_command(desired_heading, env)
            else:
                corrected_yaw_rate = yaw_cmd

            env.commands[:] = torch.tensor([[vx, vy, corrected_yaw_rate, 0.0]], device=env.device)

        # Continue as normal
        actions = actor_critic.act_inference(obs)

        lin_vel = env.root_states[:, 7:10]
        ang_vel = env.root_states[:, 10:13]

        if correct_lateral:
            lin_vel[:, 0] *= 0.0  # Zero forward drift
            ang_vel[:, 2] *= 0.0  # Zero yaw rotation
        else:
            lin_vel[:, 1] *= 0.9  # Smoothly reduce lateral drift
            ang_vel[:, 2] *= 0.9  # Smoothly reduce yaw drift
            target_vx = vx
            lin_vel[:, 0] = 0.9 * lin_vel[:, 0] + 0.1 * target_vx
            min_vx = 0.2 * np.sign(target_vx) if target_vx != 0 else 0.0
            lin_vel[:, 0] = torch.clamp(lin_vel[:, 0], min=min_vx)

        # Push corrected velocities back to sim
        env.root_states[:, 7:10] = lin_vel
        env.root_states[:, 10:13] = ang_vel
        gym_root_states_raw = env.gym.acquire_actor_root_state_tensor(env.sim)
        gym_root_states = gymtorch.wrap_tensor(gym_root_states_raw).view(-1, 13)
        gym_root_states.copy_(env.root_states)
        env.gym.set_actor_root_state_tensor(env.sim, gym_root_states_raw)


        
        # Adjust yaw
        corrected_yaw_rate = adjust_yaw_command(desired_heading, env)
        env.commands[:, 2] = corrected_yaw_rate
        actions = actor_critic.act_inference(obs)


        # Collect data **before** taking the step
        feet_contact = env.feet_state.cpu().numpy() if hasattr(env, "feet_state") else env.contact_forces.cpu().numpy()
        actions_scaled = actions * env.cfg.control.action_scale  # Apply scaling factor
        motor_torque = actions_scaled.detach().cpu().numpy() # Commanded torques (sent by the RL policy, desired torques output by the learned policy)
        sensor_torque = env.torques.detach().cpu().numpy()
        
        # Store in history
        motor_torque_history.append(motor_torque.tolist())
        sensor_torque_history.append(sensor_torque.tolist())
        com_history.append(com_pos.tolist())  
        feet_contact_history.append(feet_contact.tolist())
        joint_angles = env.dof_pos.cpu().numpy() if hasattr(env, "dof_pos") else np.zeros((env.num_envs, env.num_dof))
        joint_angle_history.append(joint_angles.tolist())

        zmp_x, zmp_y, support_x, support_y = env._compute_zmp()
        if zmp_x is None or zmp_y is None:
            return torch.tensor(0.0, device=env.device)
        zmp_positions = np.stack([zmp_x.cpu().numpy(), zmp_y.cpu().numpy()], axis=1)
        zmp_history.append(zmp_positions.tolist())
        support_polygon_history.append([support_x, support_y])

        com_x, com_y, com_z = env._compute_com()
        com_x = com_x.item() if isinstance(com_x, torch.Tensor) else com_x
        com_y = com_y.item() if isinstance(com_y, torch.Tensor) else com_y
        com_z = com_z.item() if isinstance(com_z, torch.Tensor) else com_z
        com_positions = np.array([com_x, com_y, com_z]).reshape(1, 3).tolist()
        com_history.append(com_positions)


        # Take a step
        obs, _, rews, dones, infos = env.step(actions.detach())

        # ✅ Log stiffness and damping (only if using variable stiffness)
        if hasattr(env, "cfg") and getattr(env.cfg.control, "variable_stiffness", False):
            if hasattr(env, "curr_stiffness") and hasattr(env, "curr_damping"):
                env.stiffness_log.append(env.curr_stiffness.detach().cpu().clone())
                env.damping_log.append(env.curr_damping.detach().cpu().clone())


        if step_idx % 1 == 0:
            ankle_height_when_contact = 0.07  # Approximate (could refine if needed)
            left_foot_z = env.feet_pos[:, 0, 2]  # Left foot
            right_foot_z = env.feet_pos[:, 1, 2]  # Right foot
            left_clearance = left_foot_z - ankle_height_when_contact
            right_clearance = right_foot_z - ankle_height_when_contact
            left_clearance = torch.clamp(left_clearance, min=0.0)
            right_clearance = torch.clamp(right_clearance, min=0.0)
            # print(f"Left swing clearance: {left_clearance.item():.3f} m, Right swing clearance: {right_clearance.item():.3f} m")
        step_idx += 1


        # 🔍 Access LSTM hidden states after action inference
        hidden_states, _ = actor_critic.get_hidden_states()
        if isinstance(hidden_states, tuple):  # LSTM
            hx, cx = hidden_states
            hx_np = hx[0].detach().cpu().numpy()  # [layer, batch, dim]
            cx_np = cx[0].detach().cpu().numpy()
            lstm_hidden_history.append(hx_np.tolist())
            lstm_cell_history.append(cx_np.tolist())
            # print("✅ LSTM Hidden Step:", hx_np.shape)
            # print("✅ LSTM Cell Step:", cx_np.shape)
        else:  # GRU or simple RNN
            hx = hidden_states
            hx_np = hx[0].detach().cpu().numpy()
            lstm_hidden_history.append(hx_np.tolist())
            print("✅ GRU Hidden Step:", hx_np.shape)



        if JOYSTICK_GUI:
            app.processEvents()

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

            frame = env.gym.get_camera_image(env.sim, env.envs[0], cam2_handle, gymapi.IMAGE_COLOR)

            if frame is not None:
                frame = np.frombuffer(frame, dtype=np.uint8).reshape(cam2_props.height, cam2_props.width, 4)[..., :3]
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if JOYSTICK_GUI:
                    gui_screenshot = gui.grab().toImage()
                    gui_screenshot = gui_screenshot.convertToFormat(4)  # Convert to 32-bit RGBA
                    gui_width, gui_height = gui_screenshot.width(), gui_screenshot.height()
                    ptr = gui_screenshot.bits()
                    ptr.setsize(gui_width * gui_height * 4)  # Ensure correct buffer size
                    gui_array = np.array(ptr).reshape(gui_height, gui_width, 4)
                    gui_array = cv2.cvtColor(gui_array, cv2.COLOR_RGBA2BGR)
                    gui_target_width = 600 
                    gui_target_height = 900  
                    gui_array = cv2.resize(gui_array, (gui_target_width, gui_target_height), interpolation=cv2.INTER_LINEAR)
                    frame_height, frame_width = frame.shape[:2]
                    gui_x = frame_width - gui_target_width - 10  # Right side, 10px padding
                    gui_y = 12  # Bottom side, 10px padding
                    combined_frame = frame.copy()
                    combined_frame[gui_y:gui_y+gui_target_height, gui_x:gui_x+gui_target_width] = gui_array
                else:
                    combined_frame = frame  # No GUI, just the raw frame
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use H.264 (better compression and quality)
                    fps = 30  # Use standard playback rate
                    video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (cam2_props.width, cam2_props.height))
                time_spent = time.time() - last_frame_time
                sleep_time = max(1.0 / fps - time_spent, 0)
                time.sleep(sleep_time)
                last_frame_time = time.time()
                video_writer.write(combined_frame)
                time_spent = time.time() - last_frame_time
                # Target delay per frame for real-time playback (e.g., 30 FPS → 1/30 sec = 0.033s)
                target_frame_duration = 1.0 / fps  # fps = 30
                sleep_time = max(target_frame_duration - time_spent, 0)
                time.sleep(sleep_time)
                last_frame_time = time.time()


        if current_time - start_time >= max_duration:
            print("⏳ Stopping simulation after 30 seconds.")

            if current_time - start_time > 1.0:
                elapsed_time = env.gym.get_elapsed_time(env.sim)
                sim_time = env.gym.get_sim_time(env.sim)
                print(f"⏱️ Real-time ratio: {sim_time / elapsed_time:.2f}x")

            if len(com_history) > 0:
                log_data = {
                    "com_history": com_history,
                    "motor_torque_history": motor_torque_history,
                    "sensor_torque_history": sensor_torque_history,
                    "feet_contact_history": feet_contact_history,
                    "zmp_history": zmp_history,
                    "support_polygon_history": support_polygon_history,
                    "joint_angle_history": joint_angle_history,
                }

                # 💾 Save stiffness/damping logs if available
                if hasattr(env, "stiffness_log") and len(env.stiffness_log) > 0:
                    stiffness_tensor = torch.cat(env.stiffness_log, dim=0)
                    damping_tensor = torch.cat(env.damping_log, dim=0)
                    os.makedirs("logs", exist_ok=True)
                    torch.save({
                        'stiffness': stiffness_tensor,
                        'damping': damping_tensor
                    }, 'logs/stiffness_damping_log.pt')
                    print("✅ stiffness_damping_log.pt saved.")

                log_data["lstm_hidden_history"] = lstm_hidden_history
                log_data["lstm_cell_history"] = lstm_cell_history

                # ✅ Convert tensors before saving
                log_data = convert_tensors(log_data)
                temp_path = plots_path + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(log_data, f, indent=4)
                os.rename(temp_path, plots_path)  # Atomic rename to prevent corruption
                print(f"✅ Log saved to {plots_path}")

            # Cleanup before exit
            if JOYSTICK_GUI:
                app.quit()
            if hasattr(env, "viewer") and env.viewer is not None:
                env.gym.destroy_viewer(env.viewer)
            if RECORD_FRAMES:
                video_writer.release()
                cv2.destroyAllWindows()
                print(f"🎥 Video saved to {video_out_path}")
            
            sys.exit(0)  # Exit AFTER saving the log!



if __name__ == '__main__':
    args = get_args()

    # ✅ Set default URDF directory
    urdf_dir = "resources/robots/h1/urdf/"

    # ✅ If just a filename is passed, prepend the full path
    if hasattr(args, "urdf"):
        if not os.path.isabs(args.urdf):
            args.urdf = os.path.join(urdf_dir, args.urdf)
    else:
        # If --urdf is not given at all, use a default one
        args.urdf = os.path.join(urdf_dir, "h1_soft.urdf")

    args.headless = False
    play(args)
