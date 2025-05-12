# main.py
# Entry point for running RVO-based multi-agent simulations using PyRVO.
# This script sets up the environment, initializes agents and goals according to scenario type,
# runs the simulation loop with RVO-based control, records metrics, and visualizes agent movement.

from Graph import *
from pyrvo_adapter import PyRVOController
from scenario_setup import setup_scenario
from metrics_recorder import MetricsRecorder
import numpy as np
import matplotlib.colors as mcolors

# Generate a list of visually distinct colors for agent plotting
def generate_distinct_colors(n):
    base_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    np.random.seed(0)  # Seed ensures consistent color shuffling
    np.random.shuffle(base_colors)
    return base_colors[:n]

# Create a set of agents with initial positions and goals based on the specified scenario
def generate_agents(num_agents, scenario_type):
    G = Graph()
    agents = []
    colors = generate_distinct_colors(num_agents)

    # Get initial position and goal pairs from predefined scenario setup
    positions_goals = setup_scenario(scenario_type, num_agents)

    for uid, (pos, goal) in enumerate(positions_goals):
        node = Node(uid)  # Create a new agent node with unique ID
        node.setState([pos[0], pos[1], 0.0])  # Set initial state (x, y, θ)
        node.goal = goal  # Assign goal location
        agents.append(node)
        G.addNode(node, color=colors[uid])  # Add agent node to graph with a color

    return G, agents

# Main simulation logic
if __name__ == '__main__':
    num_agents = 40
    scenario_type = "circle"  # Available: "circle", "circle_with_obstacles", etc.

    # Generate the agent graph and initialize agents
    G, agents = generate_agents(num_agents, scenario_type)

    # Optional: draw obstacles if the scenario includes them
    if scenario_type == "circle_with_obstacles":
        G.draw_obstacles()

    # Initialize the RVO controller and the metrics recorder
    rvo_controller = PyRVOController(agents, graph=G)
    metrics = MetricsRecorder(agents)

    # Define the control function used by each agent to update velocity
    def control_fn(agent):
        rvo_controller.update_from_nodes(agents)
        rvo_controller.step()
        metrics.record_step()
        return rvo_controller.get_velocity(agent.uid)

    # Assign the control function to each agent
    for node in agents:
        node.control_function = control_fn

    # Start the simulation
    print(f"Starting PyRVO-based multi-agent simulation: {scenario_type} with {num_agents} agents.")
    G.run()
    G.setupAnimation()
    G.stop()

    # Print final performance metrics after simulation
    metrics.print_summary()
    print("Simulation ended.")
