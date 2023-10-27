import open3d as o3d
import numpy as np

# Load the NPZ file
data = np.load('/home/user/RL_TM5_900_pybullet/RL_plain/npz_data/object_level/expert.npz')
check_pc_state = True
check_reward = True

if check_pc_state:
    # Extract the pc_state data from the NPZ file
    pc_state_array = data['pc_state']
    next_pc_state_array = data['next_pc_state']
    reward_array = data['reward']
    done_array = data['done']


    # Iterate through each pc_state and visualize it
    for i in range(len(pc_state_array)):
        num_points = len(pc_state_array[i])
        green_color = [0, 1, 0]  # RGB color for green
        green_colors = np.tile(green_color, (num_points, 1))  # Create an array of green_color repeated for each point
        red_color = [1, 0, 0]  # RGB color for red
        red_colors = np.tile(red_color, (num_points, 1))  # Create an array of green_color repeated for each poin
        # Create an Open3D PointCloud object from the current pc_state
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_state_array[i][:, :3])
        pcd.colors = o3d.utility.Vector3dVector(green_colors)
        next_pcd = o3d.geometry.PointCloud()
        next_pcd.points = o3d.utility.Vector3dVector(next_pc_state_array[i][:, :3])
        next_pcd.colors = o3d.utility.Vector3dVector(red_colors)
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # Visualize the PointCloud
        o3d.visualization.draw_geometries([pcd] + [next_pcd] + [axis_pcd])

        print(f"reward: {reward_array[i]}")
        print(f"done: {done_array[i]}")