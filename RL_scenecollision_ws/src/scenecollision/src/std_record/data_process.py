import numpy as np

# Function to read data from a file and convert to a 2D NumPy array
def read_data_from_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    
    # Convert each line to a NumPy array
    data_arrays = [np.fromstring(line.strip('[]\n'), sep=' ') for line in data if line.strip()]
    return np.array(data_arrays)

# Read data from both files
file1_data = read_data_from_file('gripper_std.txt')
file2_data = read_data_from_file('elbow_std.txt')


# Compute the mean of the corresponding elements
gripper_data = np.mean(file1_data, axis=0)
elbow_data = np.mean(file2_data, axis=0)

print(f"gripper_data: {gripper_data}")
print(f"elbow_data: {elbow_data}")
