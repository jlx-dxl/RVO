from Node import *
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

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
        self.obstacles = []  # <-- 仍然保留但不默认画
        self.anim = None

    def draw_obstacles(self):
        # 添加中心障碍物（以 4 个灰色方块表示）
        size = 2.0
        centers = [(-2, 2), (2, 2), (-2, -2), (2, -2)]
        for (cx, cy) in centers:
            rect = patches.Rectangle((cx - size/2, cy - size/2), size, size,
                                     linewidth=1, edgecolor='r', facecolor='gray', alpha=0.5)
            self.ax.add_patch(rect)
            self.obstacles.append(((cx - size/2, cy - size/2), (cx + size/2, cy + size/2)))  # (xmin, ymin), (xmax, ymax)

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
        return self.agent_dots + self.agent_trails + self.start_markers + self.goal_markers
