#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ---------------------------
# Helper functions
# ---------------------------
R = 6371  # Earth radius in km

def latlon_to_xyz(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return np.array([x, y, z])

def xyz_to_latlon(x, y, z):
    lat = np.degrees(np.arcsin(z / R))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon

# Linear interpolation between points
def interpolate_path(points, steps_per_segment=20):
    path = []
    for i in range(len(points)-1):
        start, end = np.array(points[i]), np.array(points[i+1])
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            path.append(start*(1-t) + end*t)
    path.append(points[-1])
    return np.array(path)

# ---------------------------
# Example data
# ---------------------------
# Cities: lat, lon
cities = {
    "A": [40.7128, -74.0060],  # NYC
    "B": [51.5074, -0.1278],   # London
    "C": [35.6895, 139.6917],  # Tokyo
}

# Local delivery points for each city (2D Cartesian local map)
local_map = {
    "A": np.array([[0,0],[1,2],[3,1],[4,3]]),
    "B": np.array([[0,0],[2,1],[1,3],[3,3]]),
    "C": np.array([[0,0],[2,2],[3,1],[4,2]])
}

# Convert city lat/lon to xyz
city_xyz = {k: latlon_to_xyz(*v) for k,v in cities.items()}

# Define global plane routes: NYC -> London -> Tokyo
plane_route = ["A","B","C"]
plane_path = interpolate_path([city_xyz[c] for c in plane_route], steps_per_segment=50)

# Define trucks: simple route in local map of each city
truck_paths = {c: interpolate_path(local_map[c], steps_per_segment=15) for c in local_map}

# ---------------------------
# Visualization
# ---------------------------
fig = plt.figure(figsize=(12,6))
ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(122)

# Globe wireframe
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
x = R * np.outer(np.cos(u), np.sin(v))
y = R * np.outer(np.sin(u), np.sin(v))
z = R * np.outer(np.ones_like(u), np.cos(v))
ax3d.plot_wireframe(x, y, z, color='grey', alpha=0.3)

# Static city points
for name, xyz in city_xyz.items():
    ax3d.scatter(*xyz, s=80, c='red')
    ax3d.text(*xyz, name)

ax3d.set_xlim(-R,R)
ax3d.set_ylim(-R,R)
ax3d.set_zlim(-R,R)
ax3d.set_title("Global Plane Routes")

# Static local map
for c, points in local_map.items():
    ax2d.scatter(points[:,0], points[:,1], label=f"City {c}")
ax2d.set_title("Local Truck/Courier Map")
ax2d.legend()
ax2d.grid(True)

# Animated lines
plane_line, = ax3d.plot([], [], [], lw=2, c='blue', alpha=1.0)
truck_lines = {c: ax2d.plot([],[],lw=2)[0] for c in local_map}

# Fade trails (store previous positions)
plane_trail = []
truck_trails = {c: [] for c in local_map}
fade_steps = 3

# ---------------------------
# Animation
# ---------------------------
def init():
    plane_line.set_data([], [])
    plane_line.set_3d_properties([])
    for line in truck_lines.values():
        line.set_data([], [])
    return [plane_line]+list(truck_lines.values())

def update(frame):
    # Global plane
    current_xyz = plane_path[:frame+1]
    plane_line.set_data(current_xyz[:,0], current_xyz[:,1])
    plane_line.set_3d_properties(current_xyz[:,2])
    plane_trail.append(current_xyz[-1])
    # Fade previous trails
    for i, p in enumerate(plane_trail):
        if i <= len(plane_trail) - fade_steps:
            plane_trail[i] = None
    # Trucks
    for c, path in truck_paths.items():
        idx = min(frame, len(path)-1)
        truck_lines[c].set_data(path[:idx+1,0], path[:idx+1,1])
        truck_trails[c].append(path[idx])
        # Fade
        if len(truck_trails[c]) > fade_steps:
            truck_trails[c].pop(0)
    return [plane_line]+list(truck_lines.values())

frames = max(len(plane_path), max(len(p) for p in truck_paths.values()))
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                              blit=True, interval=100, repeat=False)

plt.show()
