import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to read paths from a text file
def read_paths_from_txt(file_path):
    paths = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_path = []
        for line in lines:
            if line.startswith('Path'):
                if current_path:
                    paths.append(current_path)
                    current_path = []
            elif line.strip() and not line.startswith('Timestamp'):
                point = tuple(map(float, line.strip('()\n').split(', ')))
                current_path.append(point)
        if current_path:  # Append the last path if not empty
            paths.append(current_path)
    return paths

# Read paths from the text file
file_path = 'path_record.txt'
paths = read_paths_from_txt(file_path)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each path with waypoints linked together
for path in paths:
    path_x, path_y, path_z = zip(*path)
    ax.plot(path_x, path_y, path_z, marker='o')  # Plot waypoints
    ax.plot(path_x, path_y, path_z, linestyle='-', marker='')  # Connect waypoints

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Visualization of Paths with Linked Waypoints')

# Show the plot
plt.show()