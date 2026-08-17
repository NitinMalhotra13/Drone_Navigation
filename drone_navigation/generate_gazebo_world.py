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
    img_resized = img.resize((129, 129), Image.Resampling.BILINEAR)
    
    heightmap_png_path = os.path.join(models_dir, "terrain_heightmap.png")
    img_resized.save(heightmap_png_path)
    print(f"[OK] Saved Gazebo heightmap image to: {heightmap_png_path}")
    
    # 2. Extract static obstacles
    static_obstacles_xml = ""
    if os.path.exists(static_path):
        static_obs = np.load(static_path)
        tx, ty, tz = static_obs.shape
        
        obstacle_id = 0
        # Group obstacle cells vertically to form cylinders/boxes
        for x in range(tx):
            for y in range(ty):
                z_indices = np.where(static_obs[x, y, :])[0]
                if len(z_indices) > 0:
                    z_start = float(z_indices[0])
                    z_end = float(z_indices[-1] + 1)
                    height = z_end - z_start
                    center_z = z_start + (height / 2.0)
                    
                    # Convert to Gazebo coordinates
                    # (x, y) map to center of cell: (x + 0.5, y + 0.5)
                    gx = x + 0.5
                    gy = y + 0.5
                    
                    # Generate a cylindrical model for each obstacle column (trees/rocks)
                    static_obstacles_xml += f"""
    <model name='obstacle_{obstacle_id}'>
      <static>1</static>
      <pose>{gx} {gy} {center_z} 0 0 0</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry>
            <cylinder>
              <radius>0.45</radius>
              <length>{height}</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name='visual'>
          <geometry>
            <cylinder>
              <radius>0.45</radius>
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
      </link>
    </model>"""
                    obstacle_id += 1
        print(f"[OK] Grouped {obstacle_id} vertical obstacle cylinders for the Gazebo world.")
        
    # 3. Compile Gazebo World file
    world_content = f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="drone_coverage_world">
    <physics name="default_physics" default="1" type="ode">
      <max_step_size>0.01</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>100</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>50</iters>
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
