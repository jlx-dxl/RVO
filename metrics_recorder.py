# metrics_recorder.py
# Implements a utility class for logging and summarizing simulation performance metrics.
# Tracks steps, collisions, goal completions, and path efficiency of agents during RVO simulation.

import numpy as np
import time

class MetricsRecorder:
    def __init__(self, agents, radius=0.3):
        # Initialize metrics tracking for a list of agents
        self.agents = agents                           # List of all agent objects
        self.radius = radius                           # Agent radius (used for collision detection)
        self.step_count = 0                            # Total simulation steps recorded
        self.total_step_time = 0.0                     # Cumulative time spent per step
        self.collision_events = 0                      # Number of detected near-collisions
        self.start_positions = {a.uid: np.array(a.state[:2]) for a in agents}  # Initial positions
        self.arrival_times = {}                        # Step index at which each agent reached goal
        self.goal_threshold = 0.3                      # Distance threshold to consider goal reached
        self.reached_goal = set()                      # Set of agent UIDs that reached their goals

    # Call once per simulation step to update metrics
    def record_step(self):
        start_time = time.time()

        # Detect pairwise collisions between agents (based on proximity)
        positions = [a.state[:2] for a in self.agents]
        N = len(positions)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 2 * self.radius:
                    self.collision_events += 1

        # Check if any agents have newly reached their goal
        for a in self.agents:
            if a.uid in self.reached_goal:
                continue
            if np.linalg.norm(a.state[:2] - a.goal) < self.goal_threshold:
                self.arrival_times[a.uid] = self.step_count
                self.reached_goal.add(a.uid)

        self.total_step_time += time.time() - start_time
        self.step_count += 1

    # Compute a dictionary summary of all metrics tracked so far
    def summarize(self):
        N = len(self.agents)
        collision_rate = self.collision_events / self.step_count if self.step_count else 0

        # Compute path efficiency: actual distance traveled / straight-line distance
        path_efficiencies = []
        for a in self.agents:
            start = self.start_positions[a.uid]
            end = a.goal
            straight = np.linalg.norm(end - start)
            actual = np.linalg.norm(a.state[:2] - start)
            if straight > 1e-6:
                path_efficiencies.append(actual / straight)

        avg_path_eff = np.mean(path_efficiencies) if path_efficiencies else 0
        avg_step_time = self.total_step_time / self.step_count if self.step_count else 0

        summary = {
            "steps": self.step_count,
            "collisions": self.collision_events,
            "collision_rate": collision_rate,
            "avg_path_efficiency": avg_path_eff,
            "avg_step_time_sec": avg_step_time,
            "goals_reached": len(self.reached_goal)
        }

        return summary

    # Print summary statistics to the console in a formatted report
    def print_summary(self):
        summary = self.summarize()
        print("\n===== METRICS SUMMARY =====")
        for k, v in summary.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
