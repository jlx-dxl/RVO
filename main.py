from Graph import *
from pyrvo_adapter import PyRVOController
from scenario_setup import setup_scenario
from metrics_recorder import MetricsRecorder
import numpy as np
import matplotlib.colors as mcolors

# ==== 生成独特颜色 ====
def generate_distinct_colors(n):
    base_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    np.random.seed(0)
    np.random.shuffle(base_colors)
    return base_colors[:n]

# ==== 创建 agent 并设置初始状态与控制器 ====
def generate_agents(num_agents, scenario_type):
    G = Graph()
    agents = []
    colors = generate_distinct_colors(num_agents)

    # 使用统一场景函数获取起点和终点
    positions_goals = setup_scenario(scenario_type, num_agents)

    for uid, (pos, goal) in enumerate(positions_goals):
        node = Node(uid)
        node.setState([pos[0], pos[1], 0.0])
        node.goal = goal
        agents.append(node)
        G.addNode(node, color=colors[uid])

    return G, agents

# ==== 主函数入口 ====
if __name__ == '__main__':
    num_agents = 30
    scenario_type = "circle_with_obstacles"

    G, agents = generate_agents(num_agents, scenario_type)

    if scenario_type == "circle_with_obstacles":
        G.draw_obstacles()

    rvo_controller = PyRVOController(agents, graph=G)
    metrics = MetricsRecorder(agents)

    def control_fn(agent):
        rvo_controller.update_from_nodes(agents)
        rvo_controller.step()
        metrics.record_step()
        return rvo_controller.get_velocity(agent.uid)

    for node in agents:
        node.control_function = control_fn

    # 启动仿真
    print(f"Starting PyRVO-based multi-agent simulation: {scenario_type} with {num_agents} agents.")
    G.run()
    G.setupAnimation()
    G.stop()

    # 输出指标
    metrics.print_summary()
    print("Simulation ended.")
