import numpy as np

class SampleBasedController:
    def __init__(self, agents, graph=None, time_step=0.05, max_speed=0.1, num_samples=30):
        self.time_step = time_step
        self.max_speed = max_speed
        self.num_samples = num_samples
        self.agents = agents
        self.obstacles = graph.obstacles if graph is not None else []
        self.uid_to_agent = {a.uid: a for a in agents}

    def step(self):
        for agent in self.agents:
            best_vel = np.zeros(2)
            best_cost = float("inf")

            # preferred velocity direction
            goal_vec = agent.goal - agent.state[:2]
            if np.linalg.norm(goal_vec) < 1e-2:
                agent.velocity = np.zeros(2)
                continue
            pref_vel = self.max_speed * goal_vec / np.linalg.norm(goal_vec)

            # Sample velocities on a circle
            angles = np.linspace(0, 2 * np.pi, self.num_samples, endpoint=False)
            for angle in angles:
                cand = self.max_speed * np.array([np.cos(angle), np.sin(angle)])
                next_pos = agent.state[:2] + self.time_step * cand

                if self.is_in_obstacle(next_pos):
                    continue  # invalid candidate

                # cost = distance to preferred direction
                cost = np.linalg.norm(cand - pref_vel)

                if cost < best_cost:
                    best_cost = cost
                    best_vel = cand

            agent.velocity = best_vel

        for agent in self.agents:
            agent.state[:2] += self.time_step * agent.velocity

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
            a.state = node.state.copy()
            a.goal = node.goal.copy()
