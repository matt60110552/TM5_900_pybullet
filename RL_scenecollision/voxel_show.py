import open3d as o3d
import numpy as np


def create_subspace_labels(point_cloud, target_center, voxel_size, x_range, y_range, z_range):
        # Define the subspace dimensions based on the ranges and voxel size
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range

        # Calculate the number of subspaces in each dimension
        x_subspaces = int((x_max - x_min) / voxel_size)
        y_subspaces = int((y_max - y_min) / voxel_size)
        z_subspaces = int((z_max - z_min) / voxel_size)

        # Create an empty subspace label grid
        subspace_labels = np.zeros((x_subspaces, y_subspaces, z_subspaces))


        # Be careful, the original pointcloud is 0 for obstacle, 1 for target object and 2 for manipulator
        # After this function, which consider the obstacle , manipulator and target's center only, the order 
        # will become 0 for free space, 1 for obstacles, 2 for manipulator and 3 for target's center 
        # The reason of ignoring the target' points is that 3dconv only deal with relationship between these points only,
        # not for specific object's shape or feature


        # Assign obstacle's points to corresponding subspaces
        for point in point_cloud[:-10]:
            x, y, z, label = point


            # Check if the point is within the specified range
            if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
                # Calculate the subspace indices for the point
                subspace_x = int((x - x_min) / voxel_size)
                subspace_y = int((y - y_min) / voxel_size)
                subspace_z = int((z - z_min) / voxel_size)
                subspace_labels[subspace_x, subspace_y, subspace_z] += 1
            # Threshold the subspace labels: 1 for subspace with enough points, 0 for others
        threshold_value = 3  # Set a threshold for the number of points in a subspace
        subspace_labels[subspace_labels < threshold_value] = 0
        subspace_labels[subspace_labels >= threshold_value] = 1


        # Assign arm's points to corresponding subspaces
        for point in point_cloud[-10:]:
            x, y, z, label = point
            if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
                # Calculate the subspace indices for the point
                subspace_x = int((x - x_min) / voxel_size)
                subspace_y = int((y - y_min) / voxel_size)
                subspace_z = int((z - z_min) / voxel_size)
                subspace_labels[subspace_x, subspace_y, subspace_z] = label

        # Assign targetcenter to corresponding subspaces
        (center_x, center_y, center_z) = target_center
        if x_min <= center_x <= x_max and y_min <= center_y <= y_max and z_min <= center_z <= z_max:
            center_x_idx = int((center_x - x_min) / voxel_size)
            center_y_idx = int((center_y - y_min) / voxel_size)
            center_z_idx = int((center_z - z_min) / voxel_size)
        else:
            raise ValueError("The target center is not in the working space")

        subspace_labels[center_x_idx, center_y_idx, center_z_idx] = 3

        return subspace_labels

if __name__ == "__main__":
     # Example point cloud (replace this with your actual point cloud data)
    point_cloud_data = np.random.rand(100, 3)  # Example random point cloud with 100 points
    
    data = np.load('npz_data/scene_level/expert.npz')
    # Access the arrays stored in the npz file
    for key in data.files:
        array = data[key]
        print(f"Array '{key}'")
    # Define parameters
    voxel_size = 0.1  # Voxel size of 10cm
    x_range = (0.3, 1.3)
    y_range = (-0.5, 0.5)
    z_range = (0, 1)


    for idx in range(10):
        point_cloud_data = data["scene_pointcloud"][idx]


        target_center = data["goal_pos"][idx]

        # Create the subspace labels
        subspace_labels_result = create_subspace_labels(point_cloud_data, target_center, voxel_size, x_range, y_range, z_range)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])

        # Visualize the point cloud
        # o3d.visualization.draw_geometries([point_cloud])


        # Convert the 3D array to a point cloud with a feature column
        points = []
        for i in range(subspace_labels_result.shape[0]):
            for j in range(subspace_labels_result.shape[1]):
                for k in range(subspace_labels_result.shape[2]):
                    if subspace_labels_result[i, j, k] != 0:  # Ignore points with feature 0 (free space)
                        points.append([i, j, k, subspace_labels_result[i, j, k]])

        point_cloud = np.array(points)  # Convert to NumPy array
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])  # Extract coordinates
        pcd.colors = o3d.utility.Vector3dVector([
            [1.0, 0.0, 0.0] if f == 1 else [0.0, 1.0, 0.0] if f == 2 else [0.0, 0.0, 1.0] for f in point_cloud[:, 3]
        ])  # Assign colors based on feature (1: Red, 2: Green, 3: Blue)

        # Visualize the point cloud with colors
        o3d.visualization.draw_geometries([pcd])