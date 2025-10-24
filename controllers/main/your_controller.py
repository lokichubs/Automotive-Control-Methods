
# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

class PID:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0

    def reset(self):
        self.integral = 0
        self.prev_error = 0

    def update(self, setpoint, measurement):
        error = setpoint - measurement
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

        # Iter 1: 0.1, 0.0, 0.0
        # Iter 2: 0.11435888100000004, 0.0, 0.0178966525
        # Iter 3: reducing Kp a bit to reduce oscillations
        self.lateral_pid = PID(Kp=0.1, Ki=0.0, Kd=0.0178966525, dt=0.032) 

        # Iter 1: 29.91, 1.36, 0.03
        # Iter 2: 27.580149049219923, 0.4822354299071474, 0.0734521331616762
        self.longitudinal_pid = PID(27.580149049219923, 0.4822354299071474, 0.0734521331616762, 0.032)

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # ---------------|Lateral Controller|-------------------------
        # closest node
        cross_track_error, idx = closestNode(X, Y, trajectory)
        closest_node = trajectory[idx]
        next_node = trajectory[(idx + 1) % len(trajectory)]  # safe wrap-around

        # Path tangent vector
        t_dx = next_node[0] - closest_node[0]
        t_dy = next_node[1] - closest_node[1]

        # Vector from path to car
        v_dx = X - closest_node[0]
        v_dy = Y - closest_node[1]

        # Signed cross-track error
        signed_cte = np.sign(t_dx * v_dy - t_dy * v_dx) * cross_track_error

        # PID control
        delta = self.lateral_pid.update(0, signed_cte)
        delta = clamp(delta, -np.pi/6, np.pi/6)

        # ---------------|Longitudinal Controller|-------------------------
        # Many of the attempt changes were made by adjusting speed
        desired_speed = 10.5
        current_speed = np.sqrt(xdot**2 + ydot**2)
        F = self.longitudinal_pid.update(desired_speed, current_speed)
        F = clamp(F, 0, 15736)

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
