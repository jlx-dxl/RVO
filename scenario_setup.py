import numpy as np

def setup_scenario(scenario_type, num_agents):
    agents_info = []

    if scenario_type in ["circle", "circle_with_obstacles"]:
        radius = 4.0
        angles = np.linspace(0, 2 * np.pi, num_agents, endpoint=False)
        for theta in angles:
            pos = radius * np.array([np.cos(theta), np.sin(theta)])
            goal = -pos
            agents_info.append((pos, goal))

    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    return agents_info
