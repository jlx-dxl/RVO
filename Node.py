from threading import Thread
import numpy as np
import time

class Node(Thread):
    def __init__(self, uid):
        Thread.__init__(self)
        self.uid = uid
        self.state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.velocity = np.array([0.0, 0.0])    # 当前速度
        self.goal = np.array([0.0, 0.0])        # 静态目标
        self.done = False
        self.nominaldt = 0.05
        self.control_function = None

    def setState(self, s):
        self.state = np.array(s)

    def getState(self):
        return self.state

    def terminate(self):
        self.done = True

    def run(self):
        while not self.done:
            start = time.time()
            self.systemdynamics()
            end = time.time()
            time.sleep(max(self.nominaldt - (end - start), 0))

    def systemdynamics(self):
        if self.control_function is not None:
            velocity = self.control_function(self)
            self.velocity = velocity
            norm = np.linalg.norm(velocity)
            if norm > 1e-6:
                velocity = velocity / norm
                self.state[:2] += self.nominaldt * velocity
