# Graph.py
# Defines the Graph class used to represent and visualize a multi-agent system.
# It manages a collection of agents (nodes), visualizes their motion with matplotlib animation,
# and optionally renders rectangular obstacles and saves simulation frames as images.

from Node import *
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import os

class Graph:
    def __init__(self):
        # Initialize the graph structure and plotting environment
        self.Nv = 0                         # Number of nodes
        self.V = []                         # List of nodes
        self.colors = []                    # Color for each node (for plotting)
        self.traces = []                    # List of past positions (for trajectory visualization)
        self.fig = plt.figure()             # Matplotlib figure object
        self.ax = plt.axes(xlim=(-6, 6), ylim=(-6, 6))  # Axes with fixed limits
        self.ax.set_aspect('equal', 'box')

        # Matplotlib line/marker handles for agent rendering
        self.agent_dots = []
        self.agent_trails = []
        self.start_markers = []
        self.goal_markers = []

        self.obstacles = []                 # List of rectangular obstacles as (min_xy, max_xy)
        self.anim = None                    # Animation handle

        self.frame_output_dir = "frames_circle_with_obstacles_30"  # Output folder for saved frames
        self.frame_index = 0                # Current frame number for saving

    # Draw four fixed-size square obstacles at predefined locations
    def draw_obstacles(self):
        size = 1.5
        centers = [(-1.5, 1.5), (1.5, 1.5), (-1.5, -1.5), (1.5, -1.5)]
        for (cx, cy) in centers:
            rect = patches.Rectangle((cx - size/2, cy - size/2), size, size,
                                     linewidth=1, edgecolor='r', facecolor='gray', alpha=0.5)
            self.ax.add_patch(rect)
            self.obstacles.append(((cx - size/2, cy - size/2), (cx + size/2, cy + size/2)))

    # Add a new agent (node) to the graph, including all visual markers
    def addNode(self, n, color):
        self.V.append(n)
        self.Nv += 1
        self.colors.append(color)
        self.traces.append([n.state[:2].copy()])

        # Initialize plotting elements for the new agent
        dot, = self.ax.plot([], [], 'o', color=color)       # Current position
        trail, = self.ax.plot([], [], '-', lw=1, color=color)  # Trajectory
        start_marker, = self.ax.plot([], [], 's', color=color) # Start position marker
        goal_marker, = self.ax.plot([], [], '*', color=color)  # Goal position marker
        self.agent_dots.append(dot)
        self.agent_trails.append(trail)
        self.start_markers.append(start_marker)
        self.goal_markers.append(goal_marker)

    # Start all node threads (each node runs its own behavior)
    def run(self):
        for i in range(self.Nv):
            self.V[i].start()

    # Stop all node threads and wait for them to finish
    def stop(self):
        for i in range(self.Nv):
            self.V[i].terminate()
        for i in range(self.Nv):
            self.V[i].join()

    # Launch the matplotlib animation and show the GUI
    def setupAnimation(self):
        if not os.path.exists(self.frame_output_dir):
            os.makedirs(self.frame_output_dir)
        self.anim = animation.FuncAnimation(self.fig, self.animate, interval=100, blit=False)
        plt.show()

    # Animation update function: refresh agent states and save frames
    def animate(self, i):
        for idx, node in enumerate(self.V):
            pos = node.state[:2]
            self.traces[idx].append(pos.copy())
            xs = [p[0] for p in self.traces[idx]]
            ys = [p[1] for p in self.traces[idx]]

            self.agent_dots[idx].set_data([pos[0]], [pos[1]])
            self.agent_trails[idx].set_data(xs, ys)
            self.start_markers[idx].set_data([xs[0]], [ys[0]])
            self.goal_markers[idx].set_data([node.goal[0]], [node.goal[1]])

        # Save current frame as an image file
        frame_path = os.path.join(self.frame_output_dir, f"frame_{self.frame_index:04d}.png")
        self.fig.savefig(frame_path)
        self.frame_index += 1

        return self.agent_dots + self.agent_trails + self.start_markers + self.goal_markers
