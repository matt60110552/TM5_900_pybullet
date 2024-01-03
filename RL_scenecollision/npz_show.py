import open3d as o3d
import numpy as np

# Load the NPZ file
data = np.load('/home/user/RL_TM5_900_pybullet/RL_approach/npz_data/scene_level/expert.npz')
check_pc_state = True
check_reward = True

if check_pc_state:
    # Extract the pc_state data from the NPZ file
    scene_pointcloud_array = data['scene_pointcloud']
    next_scene_pointcloud_array = data['next_scene_pointcloud']
    reward_array = data['reward']
    done_array = data['done']
    goal_pos_array = data['goal_pos']


    # Iterate through each pc_state and visualize it
    # for i in range(len(scene_pointcloud_array)):
    #     num_points = len(scene_pointcloud_array[i])
    #     green_color = [0, 1, 0]  # RGB color for green
    #     green_colors = np.tile(green_color, (num_points, 1))  # Create an array of green_color repeated for each point
    #     print(f"green_colors: {green_colors.shape}")
    #     red_color = [1, 0, 0]  # RGB color for red
    #     red_colors = np.tile(red_color, (num_points, 1))  # Create an array of green_color repeated for each poin
    #     # Create an Open3D PointCloud object from the current pc_state
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(scene_pointcloud_array[i][:, :3])
    #     pcd.colors = o3d.utility.Vector3dVector(green_colors)
        
        
    #     next_pcd = o3d.geometry.PointCloud()
    #     next_pcd.points = o3d.utility.Vector3dVector(next_scene_pointcloud_array[i][:, :3])
    #     next_pcd.colors = o3d.utility.Vector3dVector(red_colors)
    #     axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    #     # Visualize the PointCloud
    #     # o3d.visualization.draw_geometries([pcd] + [next_pcd] + [axis_pcd])
    #     o3d.visualization.draw_geometries([pcd] + [axis_pcd])

    #     print(f"reward: {reward_array[i]}")
    #     print(f"done: {done_array[i]}")
    #     print(f"goal_pos: {goal_pos_array[i]}")

for i in range(len(scene_pointcloud_array)):
    num_points = len(scene_pointcloud_array[i])

    # Extract the color information from the fourth column of the point cloud data
    color_values = scene_pointcloud_array[i][:, 3]
    normalized_colors = (color_values - np.min(color_values)) / (np.max(color_values) - np.min(color_values))
    # Create an array of zeros with shape (num_points, 3) and set the middle element of each row to the color value
    colors = np.zeros((num_points, 3))
    colors[:, 1] = normalized_colors
    print(f"colors: {colors.shape}")
    # Create an Open3D PointCloud object from the current pc_state
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_pointcloud_array[i][:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a coordinate frame
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # Visualize the PointCloud
    o3d.visualization.draw_geometries([pcd, axis_pcd])

    print(f"reward: {reward_array[i]}")
    print(f"done: {done_array[i]}")
    print(f"goal_pos: {goal_pos_array[i]}")
        