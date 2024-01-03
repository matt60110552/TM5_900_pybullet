import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
import trimesh
import alphashape
import open3d as o3d

obs_pc = np.load('point_cloud.npy')


noise = 0.02 * np.random.randn(obs_pc.shape[0], 3)
noisy_pc = obs_pc + noise

# Concatenate the original and noisy point clouds
combined_pc = np.concatenate((obs_pc, noisy_pc), axis=0)



obs_pcd = o3d.geometry.PointCloud()
obs_pcd.points = o3d.utility.Vector3dVector(np.array(obs_pc[:, :3]))
noise_pcd = o3d.geometry.PointCloud()
noise_pcd.points = o3d.utility.Vector3dVector(np.array(noisy_pc[:, :3]))

# Create coordinate axes
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

# Visualize the point cloud with axes
o3d.visualization.draw_geometries([obs_pcd, noise_pcd, axes])







obs_alph = alphashape.alphashape(combined_pc, 8)

print(f"obs_alph: {obs_alph.vertices} {obs_alph.faces}")
# obs_alph.show()
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(*zip(*obs_alph.vertices), triangles=obs_alph.faces)
plt.show()