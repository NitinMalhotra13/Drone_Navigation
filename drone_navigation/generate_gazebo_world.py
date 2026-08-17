# drone_navigation/generate_gazebo_world.py
import os
import numpy as np
from PIL import Image

def generate_gazebo_assets():
    # Paths
    src_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(src_dir, "..", "dataset")
    models_dir = os.path.join(src_dir, "..", "models")
    worlds_dir = os.path.join(src_dir, "..", "worlds")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(worlds_dir, exist_ok=True)
    
    terrain_path = os.path.join(dataset_dir, "terrain.npy")
    static_path = os.path.join(dataset_dir, "static_obstacles.npy")
    
    if not os.path.exists(terrain_path):
        print(f"[ERROR] Missing terrain.npy at {terrain_path}. Please run generate_terrain.py first.")
        return
        
    # 1. Load terrain and create PNG heightmap (129x129 size for Gazebo Classic)
    terrain = np.load(terrain_path)
    max_height = float(terrain.max())
    
    # Scale terrain directly to [0, 255] range relative to max_height
    scaled_terrain = (terrain / max_height) * 255.0
    scaled_terrain = np.clip(scaled_terrain, 0.0, 255.0).astype(np.uint8)
    
    # Resize to 129x129 for Gazebo Classic terrain LOD alignment
    img = Image.fromarray(scaled_terrain, mode='L')
    img_resized = img.resize((129, 129), Image.BILINEAR)
    
    heightmap_png_path = os.path.join(models_dir, "terrain_heightmap.png")
    img_resized.save(heightmap_png_path)
    print(f"[OK] Saved Gazebo heightmap image to: {heightmap_png_path}")
    
    # 2. Extract static obstacles
    static_obstacles_xml = ""
    if os.path.exists(static_path):
        static_obs = np.load(static_path)
        tx, ty, tz = static_obs.shape
        
        obstacle_id = 0
        # Group obstacle cells vertically to form cylinders/boxes, adding height randomness & extra trees
        for x in range(tx):
            for y in range(ty):
                z_indices = np.where(static_obs[x, y, :])[0]
                has_tree = False
                gx, gy, height, center_z = 0.0, 0.0, 0.0, 0.0
                
                if len(z_indices) > 0:
                    z_start = float(z_indices[0])
                    # Apply random height scaling (between 0.7x and 2.0x) to existing trees
                    height = (float(z_indices[-1] + 1) - z_start) * np.random.uniform(0.7, 2.0)
                    center_z = z_start + (height / 2.0)
                    gx = x + 0.5
                    gy = y + 0.5
                    has_tree = True
                else:
                    # Procedurally increase tree density: 2.5% chance to spawn an extra tree in empty cells
                    # Exclude the starting zone (x < 10 and y < 10) to avoid blocking the drone spawning zone
                    if (x > 10 or y > 10) and np.random.random() < 0.025:
                        height = float(np.random.uniform(3.0, 9.0))
                        center_z = height / 2.0
                        gx = x + 0.5
                        gy = y + 0.5
                        has_tree = True
                
                if has_tree:
                    # Generate a realistic tree model with single simplified cylinder collision for ultra-fast physics
                    static_obstacles_xml += f"""
    <model name='obstacle_{obstacle_id}'>
      <static>1</static>
      <pose>{gx} {gy} {center_z} 0 0 0</pose>
      <link name='link'>
        <!-- Single Simplified Tree Collision for high FPS physics -->
        <collision name='collision'>
          <geometry>
            <cylinder>
              <radius>0.35</radius>
              <length>{height}</length>
            </cylinder>
          </geometry>
        </collision>
        <!-- Trunk Visual (Brown Wood) -->
        <visual name='visual_trunk'>
          <geometry>
            <cylinder>
              <radius>0.15</radius>
              <length>{height}</length>
            </cylinder>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Wood</name>
            </script>
          </material>
        </visual>
        <!-- Canopy Visual (Green Foliage Bushes) -->
        <visual name='visual_canopy'>
          <pose>0 0 {height / 2.0}</pose>
          <geometry>
            <sphere>
              <radius>0.6</radius>
            </sphere>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Green</name>
            </script>
          </material>
        </visual>
      </link>
    </model>"""
                    obstacle_id += 1
        
        # Add 20 dynamic moving obstacles (red spheres with planar move plugin)
        np.random.seed(42)
        dynamic_starts = []
        for _ in range(20):
            dx = float(np.random.uniform(15.0, 90.0))
            dy = float(np.random.uniform(15.0, 90.0))
            dynamic_starts.append((dx, dy))
            
        for i, (dx, dy) in enumerate(dynamic_starts):
            static_obstacles_xml += f"""
    <model name='dynamic_obstacle_{i}'>
      <pose>{dx} {dy} 0.5 0 0 0</pose>
      <link name='link'>
        <!-- Collision sphere -->
        <collision name='collision'>
          <geometry>
            <sphere>
              <radius>0.5</radius>
            </sphere>
          </geometry>
        </collision>
        <!-- Visual sphere (Red) -->
        <visual name='visual'>
          <geometry>
            <sphere>
              <radius>0.5</radius>
            </sphere>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Red</name>
            </script>
          </material>
        </visual>
      </link>
      <!-- Planar move plugin for dynamic velocity controller -->
      <plugin name='planar_move_{i}' filename='libgazebo_ros_planar_move.so'>
        <ros>
          <namespace>/dynamic_obstacle_{i}</namespace>
        </ros>
        <robot_base_frame>link</robot_base_frame>
      </plugin>
    </model>"""
            
        print(f"[OK] Grouped {obstacle_id} vertical obstacle cylinders for the Gazebo world and appended 20 red dynamic obstacles.")
        
    # 3. Compile Gazebo World file
    world_content = f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="drone_coverage_world">
    <physics name="default_physics" default="1" type="ode">
      <max_step_size>0.005</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>200</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>20</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- GUI User Camera Pose facing the heightmap terrain center directly -->
    <gui fullscreen='0'>
      <camera name='user_camera'>
        <pose>50.0 -15.0 50.0 0.0 0.65 1.5707</pose>
        <view_controller>orbit</view_controller>
        <projection_type>perspective</projection_type>
      </camera>
    </gui>

    <!-- Sun directional light -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Terrain Heightmap -->
    <model name="terrain_heightmap">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <heightmap>
              <uri>file://{heightmap_png_path}</uri>
              <size>100.0 100.0 {max_height}</size>
              <pos>50.0 50.0 0.0</pos>
            </heightmap>
          </geometry>
        </collision>
        <visual name="visual_heightmap">
          <geometry>
            <heightmap>
              <use_terrain_paging>false</use_terrain_paging>
              <texture>
                <diffuse>file://media/materials/textures/dirt_diffusespecular.png</diffuse>
                <normal>file://media/materials/textures/flat_normal.png</normal>
                <size>10.0</size>
              </texture>
              <uri>file://{heightmap_png_path}</uri>
              <size>100.0 100.0 {max_height}</size>
              <pos>50.0 50.0 0.0</pos>
            </heightmap>
          </geometry>
        </visual>
      </link>
    </model>

    <!-- Static obstacles -->{static_obstacles_xml}
  </world>
</sdf>
"""
    
    world_path = os.path.join(worlds_dir, "drone_coverage.world")
    with open(world_path, "w") as f:
        f.write(world_content)
    print(f"[OK] Gazebo World file written to: {world_path}")

if __name__ == "__main__":
    generate_gazebo_assets()
