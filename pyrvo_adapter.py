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
    def __init__(self, agents, time_step=0.05, neighbor_dist=2.0, time_horizon=3.0):
        self.time_step = time_step
        self.neighbor_dist = neighbor_dist
        self.time_horizon = time_horizon
        self.agents = [PyRVOAgent(a.uid, a.state[:2], a.goal) for a in agents]
        self.uid_to_agent = {a.uid: a for a in self.agents}

    def step(self):
        for a in self.agents:
            pref_vel = a.goal - a.position
            if np.linalg.norm(pref_vel) > 1e-6:
                pref_vel = a.max_speed * pref_vel / np.linalg.norm(pref_vel)
            else:
                pref_vel = np.zeros(2)

            new_vel = pref_vel.copy()

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
                    new_vel += strength * avoid_dir

            speed = np.linalg.norm(new_vel)
            if speed > a.max_speed:
                new_vel = a.max_speed * new_vel / speed

            a.velocity = new_vel

        for a in self.agents:
            a.position += self.time_step * a.velocity

    def get_velocity(self, uid):
        return self.uid_to_agent[uid].velocity

    def update_from_nodes(self, nodes):
        for node in nodes:
            a = self.uid_to_agent[node.uid]
            a.position = node.state[:2].copy()
            a.goal = node.goal.copy()
