import pybullet as p
import time
import pybullet_data
import os

# --- Configuration ---
# The name of your URDF file.
URDF_FILENAME = "dual_yam.urdf"

# Check if the URDF file exists in the current directory.
if not os.path.exists(URDF_FILENAME):
    print(f"Error: The file '{URDF_FILENAME}' was not found in this directory.")
    print("Please make sure your URDF file and this script are in the same folder.")
    exit()

# --- PyBullet Setup ---
# Start the simulation GUI. You can also use p.DIRECT for a non-graphical version.
physicsClient = p.connect(p.GUI) 

# Add a search path for PyBullet to find resources like the plane URDF.
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set the gravity for the simulation.
p.setGravity(0, 0, -9.8)

# --- Load Models ---
# Load a ground plane to have a reference surface.
planeId = p.loadURDF("plane.urdf")

# Define the starting position and orientation of the robot's base.
# Position: [x, y, z] - we'll raise it slightly off the ground.
# Orientation: [x, y, z, w] as a quaternion. [0, 0, 0, 1] is no rotation.
start_pos = [0, 0, 0.01]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])

print(f"Loading robot from '{URDF_FILENAME}'...")

# Load the robot from your URDF file.
# The `useFixedBase=True` argument makes the robot's base static,
# so it won't fall over due to gravity.
try:
    robot_id = p.loadURDF(
        URDF_FILENAME,
        start_pos,
        start_orientation,
        useFixedBase=True 
    )
    print("Robot loaded successfully!")
except Exception as e:
    print("--- FAILED TO LOAD URDF ---")
    print(f"PyBullet Error: {e}")
    print("This often happens if the STL mesh paths in the URDF are incorrect.")
    print("Please check that the 'assets' folder is in the same directory as the script.")
    p.disconnect()
    exit()


# --- Simulation Loop ---
# This loop keeps the visualization window open and responsive.
print("Starting visualization. Close the window to exit.")
try:
    while True:
        # This step is not strictly necessary if you are not running a physics sim,
        # but it's good practice and keeps the window updated.
        p.stepSimulation()
        time.sleep(1./240.) # Sleep to match a typical simulation frequency.

except p.error as e:
    # This handles the case where the user closes the GUI window.
    print("Visualization window closed by user.")

finally:
    # Always disconnect from the simulation when you're done.
    p.disconnect()
