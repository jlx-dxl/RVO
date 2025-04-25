import numpy as np

class PyRVOAgent:
    def __init__(self, uid, position, goal, radius=0.1, max_speed=1.0):
        self.uid = uid
        self.position = np.array(position)
        self.velocity = np.zeros(2)
        self.goal = np.array(goal)
        self.radius = radius
        self.max_speed = max_speed

class PyRVOController:
    def __init__(self, agents, time_step=0.05, neighbor_dist=1.0, time_horizon=3.0, graph=None, num_fallback_samples=60, max_speed=1.0):
        self.time_step = time_step
        self.neighbor_dist = neighbor_dist
        self.time_horizon = time_horizon
        self.num_fallback_samples = num_fallback_samples
        self.agents = [PyRVOAgent(a.uid, a.state[:2], a.goal) for a in agents]
        self.uid_to_agent = {a.uid: a for a in self.agents}
        self.obstacles = graph.obstacles if graph is not None else []
        self.max_speed = max_speed

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
                combined_radius = self.neighbor_dist
                if dist < combined_radius:
                    avoid_dir = -rel_pos / dist
                    strength = (combined_radius - dist) / combined_radius
                    new_vel += 1.2 * strength * avoid_dir

            # === Obstacle repulsion (soft)
            for (min_xy, max_xy) in self.obstacles:
                closest = np.clip(a.position, min_xy, max_xy)
                rel = a.position - closest
                dist = np.linalg.norm(rel)
                if dist < 1.0 and dist > 1e-6:
                    avoid_dir = rel / dist
                    strength = (1.0 - dist)
                    new_vel += 1.2 * strength * avoid_dir

            # === Predict entry into obstacle
            next_pos = a.position + self.time_step * new_vel
            if self.is_in_obstacle(next_pos):
                new_vel = self.find_fallback_velocity(a, pref_vel)

            # === Normalize
            speed = np.linalg.norm(new_vel)
            if speed > a.max_speed:
                new_vel = a.max_speed * new_vel / speed

            a.velocity = new_vel

        for a in self.agents:
            proposed_pos = a.position + self.time_step * a.velocity
            if not self.is_in_obstacle(proposed_pos):
                a.position = proposed_pos

    def find_fallback_velocity(self, agent, pref_vel):
        best_vel = np.zeros(2)
        best_cost = float("inf")
        angles = np.linspace(0, 2 * np.pi, self.num_fallback_samples, endpoint=False)
        for angle in angles:
            v = self.max_speed * np.array([np.cos(angle), np.sin(angle)])
            next_pos = agent.position + self.time_step * v
            if self.is_in_obstacle(next_pos):
                continue
            cost = np.linalg.norm(v - pref_vel)
            if cost < best_cost:
                best_cost = cost
                best_vel = v
        return best_vel

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
