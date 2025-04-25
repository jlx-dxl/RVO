import numpy as np

def setup_scenario(scenario_type, num_agents):
    agents_info = []

    if scenario_type == "circle":
        radius = 4.0
        angles = np.linspace(0, 2 * np.pi, num_agents, endpoint=False)
        for i, theta in enumerate(angles):
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            position = np.array([x, y])
            goal = -position
            agents_info.append((position, goal))

    elif scenario_type == "narrow_passage":
        grid_size = int(np.sqrt(num_agents))
        padding = 2.0
        spacing = 1.5
        offset = 4.0
        for i in range(num_agents):
            row = i // grid_size
            col = i % grid_size
            x = col * spacing - offset
            y = row * spacing - offset
            position = np.array([x, y])
            goal = -position
            agents_info.append((position, goal))

    elif scenario_type == "moving_obstacle":
        # Place agents on left side, all goals on right side
        for i in range(num_agents):
            y = np.linspace(-4, 4, num_agents)[i]
            position = np.array([-4.0, y])
            goal = np.array([4.0, y])
            agents_info.append((position, goal))

    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    return agents_info
