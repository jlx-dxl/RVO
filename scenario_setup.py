import numpy as np

def setup_scenario(scenario_type, num_agents):
    agents_info = []

    if scenario_type == "circle":
        radius = 4.0
        angles = np.linspace(0, 2 * np.pi, num_agents, endpoint=False)
        for theta in angles:
            pos = radius * np.array([np.cos(theta), np.sin(theta)])
            goal = -pos
            agents_info.append((pos, goal))

    elif scenario_type == "narrow_passage":
        grid_size = int(np.sqrt(num_agents))
        spacing = 1.5
        offset = 4.0
        for i in range(num_agents):
            row = i // grid_size
            col = i % grid_size
            pos = np.array([col * spacing - offset, row * spacing - offset])
            goal = -pos
            agents_info.append((pos, goal))

    elif scenario_type == "four_corner_cross":
        # Four groups: top-left, top-right, bottom-left, bottom-right
        agents_per_group = num_agents // 4
        spacing = 1.0
        count_per_row = int(np.ceil(np.sqrt(agents_per_group)))
        margin = 0.5
        start_offsets = {
            "top_left": (-5, 5),
            "top_right": (5, 5),
            "bottom_left": (-5, -5),
            "bottom_right": (5, -5),
        }
        goal_targets = {
            "top_left": (5, -5),
            "top_right": (-5, -5),
            "bottom_left": (5, 5),
            "bottom_right": (-5, 5),
        }

        def generate_group(offset, goal_target):
            group = []
            for i in range(agents_per_group):
                row = i // count_per_row
                col = i % count_per_row
                dx = col * spacing
                dy = row * spacing
                pos = np.array([offset[0] + dx, offset[1] - dy])
                goal = np.array(goal_target)
                group.append((pos, goal))
            return group

        for region in ["top_left", "top_right", "bottom_left", "bottom_right"]:
            agents_info.extend(generate_group(start_offsets[region], goal_targets[region]))

    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    return agents_info
