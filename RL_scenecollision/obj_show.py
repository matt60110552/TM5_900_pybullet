# import open3d as o3d
# import numpy as np

# # Path to your OBJ file
# obj_file_path = 'pointcloud_mesh.obj'

# # Load the OBJ file
# mesh = o3d.io.read_triangle_mesh(obj_file_path)

# # Create coordinate axes
# axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

# # Paint the mesh in red
# mesh.paint_uniform_color([1, 0, 0])  # Set color to red

# # Combine the OBJ mesh and coordinate axes
# combined_mesh = o3d.geometry.TriangleMesh()
# combined_mesh.vertices = o3d.utility.Vector3dVector(np.vstack([np.asarray(mesh.vertices), np.asarray(axes.vertices)]))
# combined_mesh.triangles = o3d.utility.Vector3iVector(np.vstack([np.asarray(mesh.triangles), np.asarray(axes.triangles)]))

# # Visualize the combined mesh with red-colored mesh and axes
# o3d.visualization.draw_geometries([combined_mesh])



import open3d as o3d

# Path to your OBJ file
obj_file_path = 'pointcloud_mesh.obj'

# Load the OBJ file
mesh = o3d.io.read_triangle_mesh(obj_file_path)

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])
