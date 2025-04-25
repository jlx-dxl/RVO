import numpy as np

class PyRVOAgent:
    def __init__(self, uid, position, goal, radius=0.3, max_speed=1.0):
        self.uid = uid
        self.position = np.array(position)
        self.velocity = np.zeros(2)
        self.goal = np.array(goal)
        self.radius = radius
        self.max_speed = max_speed

class PyRVOController:
    def __init__(self, agents, time_step=0.05, neighbor_dist=2.5, time_horizon=3.0, graph=None):
        self.time_step = time_step
        self.neighbor_dist = neighbor_dist
        self.time_horizon = time_horizon
        self.agents = [PyRVOAgent(a.uid, a.state[:2], a.goal) for a in agents]
        self.uid_to_agent = {a.uid: a for a in self.agents}
        self.obstacles = graph.obstacles if graph is not None else []

    def step(self):
        for a in self.agents:
            pref_vel = a.goal - a.position
            if np.linalg.norm(pref_vel) > 1e-6:
                pref_vel = a.max_speed * pref_vel / np.linalg.norm(pref_vel)
            else:
                pref_vel = np.zeros(2)

            new_vel = pref_vel.copy()

            # === Agent repulsion ===
            for b in self.agents:
                if b.uid == a.uid:
                    continue
                rel_pos = b.position - a.position
                dist = np.linalg.norm(rel_pos)
                if dist < 1e-6 or dist > self.neighbor_dist:
                    continue

                combined_radius = a.radius + b.radius
                if dist < combined_radius:
                    avoid_dir = -rel_pos / dist
                    strength = (combined_radius - dist) / combined_radius
                    new_vel += 1.0 * strength * avoid_dir

            # === Obstacle soft repulsion ===
            for (min_xy, max_xy) in self.obstacles:
                closest = np.clip(a.position, min_xy, max_xy)
                rel = a.position - closest
                dist = np.linalg.norm(rel)
                if dist < 1.0 and dist > 1e-6:
                    avoid_dir = rel / dist
                    strength = (1.0 - dist)
                    new_vel += 0.8 * strength * avoid_dir

            # === Predict entry into obstacle: block hard ===
            next_pos = a.position + self.time_step * new_vel
            if self.is_in_obstacle(next_pos):
                new_vel = np.zeros(2)

            # === Clip velocity ===
            speed = np.linalg.norm(new_vel)
            if speed > a.max_speed:
                new_vel = a.max_speed * new_vel / speed

            a.velocity = new_vel

        # === Final position update with obstacle check ===
        for a in self.agents:
            proposed_pos = a.position + self.time_step * a.velocity
            if not self.is_in_obstacle(proposed_pos):
                a.position = proposed_pos  # only move if result is valid

    def is_in_obstacle(self, pos):
        for (min_xy, max_xy) in self.obstacles:
            if all(min_xy[i] <= pos[i] <= max_xy[i] for i in range(2)):
                return True
        return False

    def get_velocity(self, uid):
        return self.uid_to_agent[uid].velocity

    def update_from_nodes(self, nodes):
        for node in nodes:
            a = self.uid_to_agent[node.uid]
            a.position = node.state[:2].copy()
            a.goal = node.goal.copy()
