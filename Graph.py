from Node import *
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import os

class Graph:
    def __init__(self):
        self.Nv = 0
        self.V = []
        self.colors = []
        self.traces = []
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-6, 6), ylim=(-6, 6))
        self.ax.set_aspect('equal', 'box')
        self.agent_dots = []
        self.agent_trails = []
        self.start_markers = []
        self.goal_markers = []
        self.obstacles = []
        self.anim = None
        self.frame_output_dir = "frames_circle_with_obstacles_30"
        self.frame_index = 0

    def draw_obstacles(self):
        size = 1.5
        centers = [(-1.5, 1.5), (1.5, 1.5), (-1.5, -1.5), (1.5, -1.5)]
        for (cx, cy) in centers:
            rect = patches.Rectangle((cx - size/2, cy - size/2), size, size,
                                     linewidth=1, edgecolor='r', facecolor='gray', alpha=0.5)
            self.ax.add_patch(rect)
            self.obstacles.append(((cx - size/2, cy - size/2), (cx + size/2, cy + size/2)))

    def addNode(self, n, color):
        self.V.append(n)
        self.Nv += 1
        self.colors.append(color)
        self.traces.append([n.state[:2].copy()])
        dot, = self.ax.plot([], [], 'o', color=color)
        trail, = self.ax.plot([], [], '-', lw=1, color=color)
        start_marker, = self.ax.plot([], [], 's', color=color)
        goal_marker, = self.ax.plot([], [], '*', color=color)
        self.agent_dots.append(dot)
        self.agent_trails.append(trail)
        self.start_markers.append(start_marker)
        self.goal_markers.append(goal_marker)

    def run(self):
        for i in range(self.Nv):
            self.V[i].start()

    def stop(self):
        for i in range(self.Nv):
            self.V[i].terminate()
        for i in range(self.Nv):
            self.V[i].join()

    def setupAnimation(self):
        if not os.path.exists(self.frame_output_dir):
            os.makedirs(self.frame_output_dir)
        self.anim = animation.FuncAnimation(self.fig, self.animate, interval=100, blit=False)
        plt.show()

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

        # 保存每帧图像
        frame_path = os.path.join(self.frame_output_dir, f"frame_{self.frame_index:04d}.png")
        self.fig.savefig(frame_path)
        self.frame_index += 1

        return self.agent_dots + self.agent_trails + self.start_markers + self.goal_markers
