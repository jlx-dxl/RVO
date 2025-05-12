# Node.py
# Defines the Node class representing an autonomous agent in the simulation.
# Each node runs as a separate thread, simulating the agent's motion toward a goal
# using velocity commands determined by a control function.

from threading import Thread
import numpy as np
import time

class Node(Thread):
    def __init__(self, uid):
        # Initialize the agent thread with a unique ID and default parameters
        super().__init__()
        self.uid = uid                                      # Unique identifier of the agent
        self.state = np.array([0.0, 0.0, 0.0])              # Agent state: [x, y, theta]
        self.velocity = np.array([0.0, 0.0])                # Current 2D velocity vector
        self.goal = np.array([0.0, 0.0])                    # Static goal position
        self.done = False                                   # Termination flag for thread loop
        self.nominaldt = 0.05                               # Desired control update timestep
        self.control_function = None                        # Control callback for computing velocity

    # Manually set the agent's state
    def setState(self, s):
        self.state = np.array(s)

    # Retrieve the agent's current state
    def getState(self):
        return self.state

    # Signal the thread to stop
    def terminate(self):
        self.done = True

    # Main thread execution loop: updates agent state using control logic
    def run(self):
        while not self.done:
            start = time.time()
            self.systemdynamics()  # Apply motion update
            end = time.time()
            time.sleep(max(self.nominaldt - (end - start), 0))  # Enforce fixed timestep

    # Apply velocity update based on control function
    def systemdynamics(self):
        if self.control_function is not None:
            velocity = self.control_function(self)  # Compute desired velocity
            self.velocity = velocity
            norm = np.linalg.norm(velocity)
            if norm > 1e-6:
                velocity = velocity / norm          # Normalize to unit direction
                self.state[:2] += self.nominaldt * velocity  # Move in direction of velocity
