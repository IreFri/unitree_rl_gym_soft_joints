
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import ctypes

# Define control model flags (choose one set at a time)
VARIABLE_STIFFNESS = True  # If True → use variable stiffness policy control
FIXED_STIFFNESS = not VARIABLE_STIFFNESS  # Automatically inferred

# If FIXED_STIFFNESS → choose soft or rigid preset
FIXED_STIFFNESS_PRESET = 'rigid'  # Options: 'soft' or 'rigid'

# choose stiffness model (parallel or series)
PARALLEL = True   # stiffness in parallel configuration
SERIE = False  # stiffness in series configuration

if VARIABLE_STIFFNESS:
    NUM_ACTIONS = 10 + 10 + 10
    NUM_OBS = 81  # 3+3+3+10+10+30+1+1+10+10=71  (stiffness and dampings are 10+10 because are repeated for the both legs)
    NUM_PRIVILEGED_OBS = 84  
else:
    NUM_ACTIONS = 10
    NUM_OBS = 41 # 3+3+3+10+10+10+1+1=41
    NUM_PRIVILEGED_OBS = 44 # 3+3+3+3+10+10+10+1+1=44 (linear velocity in plus)


class H1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.0] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,     # 0.3
           'left_ankle_joint' : -0.2,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,  # 0.3                                           
           'right_ankle_joint' : -0.2,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
        }
    
    class env(LeggedRobotCfg.env):
        num_actions = NUM_ACTIONS
        num_observations = NUM_OBS
        num_privileged_obs = NUM_PRIVILEGED_OBS
        test = True
        send_timeouts = True
        num_envs = 4096 # 4096
        env_spacing = 3.   
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 120 # episode length in seconds
        test = False

    class terrain:
        mesh_type = "plane" # "plane" for training on flat ground, "heightfield" for training on uneven
        terrain_type= "none" # SET "flat_only_debug" for debuggin on the flat ground, "none" otherwise
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 1 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        slope_threshold = 0.5
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        max_init_terrain_level = 2 # starting curriculum state
        # Here patch size
        terrain_length = 4.0
        terrain_width = 4.0
        num_rows= 48 # number of terrain rows (levels) NEED TO BE: num_envs = num_rows × num_cols × robots_per_patch
        num_cols = 48 # number of terrain cols (types)
        robots_per_patch = 1
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2] # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0] # terrain types: only discrete
        terrain_kwargs = None # Dict of arguments for selected terrain



    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
      

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.75]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 2.0 # 2.5
        # randomize_friction = True
        # friction_range = [0.1, 1.25]
        # randomize_base_mass = True
        # added_mass_range = [-1., 3.]
        # push_robots = True
        # push_interval_s = 5
        # max_push_vel_xy = 1.5


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'T'
        variable_stiffness = VARIABLE_STIFFNESS
        fixed_stiffness = FIXED_STIFFNESS
        max_stiffness = 250.0
        max_damping = 6.0

        if FIXED_STIFFNESS:
            if FIXED_STIFFNESS_PRESET == 'soft':
                fixed_stiffness_values = {
                    'stiffness': [200.0, 100.0, 80.0, 200.0, 100.0] * 2,
                    'damping':   [4.5, 4.5, 1.0, 5.0, 1.0] * 2
                }
            elif FIXED_STIFFNESS_PRESET == 'rigid':
                fixed_stiffness_values = {
                    'stiffness': [0.0] * 10,
                    'damping':   [0.0] * 10
                }
            if PARALLEL:
                fixed_stiffness_model = "parallel"
            elif SERIE:
                fixed_stiffness_model = "series"

        
        if PARALLEL:
            stiffness_model = "parallel"
        elif SERIE:
            stiffness_model = "series"

        torque_limits = {
            'hip_yaw': 60.0, 'hip_roll': 60.0, 'hip_pitch': 120.0,
            'knee': 150.0, 'ankle': 40.0, 'torso': 50.0
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 35 # For soft robot action_scale = 35
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1_normal.urdf'
        name = "h1"
        foot_name = "ankle"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        spawn_offset_z = 0.0
        collision_group = 1
        collision_mask = -1

  
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 1.00 
        contact_force_threshold = 1.0
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 60.0
            tracking_lin_vel_y = 40
            tracking_ang_vel = 15.0
            lin_vel_z = -5.0  
            ang_vel_xy = -0.5  
            orientation = -15.0  # It was -15
            base_height = -200.0  #(soft -200)
            dof_acc = -5e-6  #(soft -5e-6)
            feet_air_time = 10.0  #(soft 10)
            collision = -15.0  
            action_rate = -0.1 #(soft -0.1)
            torques = -0.0015  #(soft -0.0015)
            dof_pos_limits = -80.0  #(soft -80)
            alive = 5.0  
            hip_pos = -30.0  #(soft -10)
            contact_no_vel = -150.0  #(soft -100)
            contact = 60.0 #(soft 6)
            step_alternation = 20.0 # #(soft 20)
            ankle_torque_penalty = -0.1 #(soft -0.2) if hit the tip of the foot, lower it
            anti_hop = -10.0
            # low_swing_foot_clearance = 10.0


class H1RoughCfgPPO( LeggedRobotCfgPPO ):
    num_steps_per_env = 24 # per iteration
    save_interval = 50
    empirical_normalization = False
    class policy:
        init_noise_std = 0.2
        actor_hidden_dims = [512, 512]  # var stiff [512, 512]
        critic_hidden_dims = [512, 512] 
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 128 # var stiff 128
        rnn_num_layers = 1
        class_name = "ActorCriticRecurrent"
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 3e-4
        class_name = "PPO"
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 20000
        run_name = ''
        experiment_name = 'h1'
