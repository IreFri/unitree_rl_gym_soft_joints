
import numpy as np
from numpy.random import choice
from scipy import interpolate
from isaacgym import terrain_utils
from isaacgym import gymapi
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy.ndimage import zoom


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_envs = num_robots
        # Promote `max_height` from kwargs if available
        terrain_kwargs = getattr(cfg, 'terrain_kwargs', {}) or {}
        if "max_height" in terrain_kwargs:
            self.cfg.max_height = terrain_kwargs["max_height"]
        else:
            self.cfg.max_height = 0.05  # default fallback
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if cfg.mesh_type == "heightfield":
            self.vertical_scale = cfg.vertical_scale
            self.horizontal_scale = cfg.horizontal_scale
        else:
            self.vertical_scale = None  
            self.horizontal_scale = None

        if self.type in ["none", 'plane']:
            print("⚠️ Skipping terrain generation due to mesh_type =", self.type)
            return

        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        self.center_x = self.height_field_raw.shape[0] // 2
        self.center_y = self.height_field_raw.shape[1] // 2

        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            max_allowed_height = int(self.cfg.max_height / self.cfg.vertical_scale)  # e.g., 0.05 / 0.005 = 10
            self.height_field_raw = np.clip(self.height_field_raw, -max_allowed_height, max_allowed_height)

            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_threshold
            )    

        if self.cfg.terrain_type == "flat_only_debug":
            self.height_field_raw[:, :] = 0
        
        self.finalize_terrain()  
  
        print(f"🔧 Selected terrain: {self.cfg.terrain_kwargs}")

    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            patch_idx = k
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty, patch_idx)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_kwargs = self.cfg.terrain_kwargs.copy() if self.cfg.terrain_kwargs else {}
        terrain_type = terrain_kwargs.pop('type')        # Get terrain function name

        if terrain_type in ['plane', 'none']:
            print(f"🛬 Flat terrain selected ({terrain_type}), skipping generation.")
            return

        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale
            )

            eval(terrain_type)(terrain, **terrain_kwargs)  
            self.add_terrain_to_map(terrain, i, j)


    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.01, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            min_size_range = 0.1         # from small to large obstacles
            max_size_range = 0.2         # from moderately large to very large
            height_range    = [0.1, 0.1]       # from low to tall obstacles
            num_rects_range = [100, 200]        # from sparse to dense
            discrete_obstacles_height = height_range[0] + difficulty * (height_range[1] - height_range[0])
            num_rectangles = int(num_rects_range[0] + difficulty * (num_rects_range[1] - num_rects_range[0]))
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, min_size_range, max_size_range, num_rectangles, platform_size=0.5)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)

        terrain.height_field_raw = np.maximum(terrain.height_field_raw, 0) # To make the stones all in positive z (not inside the ground)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system (centered)
        start_x = self.center_x + (i - self.cfg.num_rows // 2) * self.length_per_env_pixels
        end_x   = start_x + self.length_per_env_pixels
        start_y = self.center_y + (j - self.cfg.num_cols // 2) * self.width_per_env_pixels
        end_y   = start_y + self.width_per_env_pixels

        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        # ✅ Correct world coordinate computation
        env_origin_x = (i + 0.5) * self.env_length - (self.cfg.num_rows * self.env_length) / 2
        env_origin_y = (j + 0.5) * self.env_width  - (self.cfg.num_cols * self.env_width)  / 2

        # ✅ Max height of subpatch
        env_origin_z = np.max(terrain.height_field_raw) * terrain.vertical_scale

        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]




    def finalize_terrain(self):
        # Flatten to (num_envs, 3) and clip to number of envs
        self.env_origins = self.env_origins.reshape(-1, 3)[:self.num_envs]  

        terrain_offset_x = -self.height_field_raw.shape[0] * self.cfg.horizontal_scale / 2
        terrain_offset_y = -self.height_field_raw.shape[1] * self.cfg.horizontal_scale / 2

        self.env_origins[:, 0] -= terrain_offset_x
        self.env_origins[:, 1] -= terrain_offset_y




def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth