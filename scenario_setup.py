# scenario_setup.py
# Provides helper function to configure initial positions and goals for agents
# based on predefined scenario types (e.g., circle, circle with obstacles).

import numpy as np

# Generate initial positions and goal destinations for a given scenario type
def setup_scenario(scenario_type, num_agents):
    agents_info = []  # List of (initial_position, goal_position) pairs for all agents

    if scenario_type in ["circle", "circle_with_obstacles"]:
        # Place agents evenly on a circle and assign opposite positions as goals
        radius = 4.0  # Radius of the circle
        angles = np.linspace(0, 2 * np.pi, num_agents, endpoint=False)

        for theta in angles:
            pos = radius * np.array([np.cos(theta), np.sin(theta)])  # Agent start position on circle
            goal = -pos  # Goal is diametrically opposite to start position
            agents_info.append((pos, goal))

    else:
        # Raise an error if the scenario type is unsupported
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    return agents_info
