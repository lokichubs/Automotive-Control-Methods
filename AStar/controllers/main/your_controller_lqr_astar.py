# Fill in the respective functions to implement the LQR optimal controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

def wrap(angle):
    """Normalize an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

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
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

        self.delT = 0.032
        self.longitudinal_pid = PID(Kp=24.58, Ki=0.482, Kd=0.1034, dt=self.delT)
        self.desired_speed = 35

        e1_max = 0.5
        e2_max = 8
        e3_max = 0.1
        e4_max = 1.4
        del_max = 0.1
        
        self.Q = np.diag([1/(e1_max**2), 1/(e2_max**2), 1/(e3_max**2), 1/(e4_max**2)])
        self.R = np.array([[1/(del_max**2)]])

        
    def design_lateral_controller_LQR(self,v_design):
        """Calculates the state-feedback gain vector K using discrete LQR."""

        lr, lf, Ca, Iz, m,delT = self.lr, self.lf, self.Ca, self.Iz, self.m,self.delT

        A = np.array([
            [0, 1, 0, 0],
            [0, -4*Ca/(m*v_design), 4*Ca/m, -2*Ca*(lf - lr)/(m*v_design)],
            [0, 0, 0, 1],
            [0, -2*Ca*(lf - lr)/(Iz*v_design), 2*Ca*(lf - lr)/Iz, -2*Ca*(lf**2 + lr**2)/(Iz*v_design)]
        ])

        B = np.array([[0], [2*Ca/m], [0], [2*Ca*lf/Iz]])

        C = np.eye(4)
        D = np.zeros((4, 1))
        Ad, Bd, _, _, _ = signal.cont2discrete((A, B, C, D), delT, method='zoh')
        P = linalg.solve_discrete_are(Ad, Bd, self.Q, self.R)
        K = np.linalg.inv(self.R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

        return np.asarray(K).reshape(-1)

    
    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot, obstacleX, obstacleY = super().getStates(timestep)

        ## -- Lateral Controller --
        cross_track_error, idx = closestNode(X, Y, trajectory)
        closest_node = trajectory[idx]
        future_steps = 180
        next_node = trajectory[(idx + future_steps) % len(trajectory)]
        psi_des = np.arctan2(next_node[1] - closest_node[1], next_node[0] - closest_node[0])
        tangent = np.array([np.cos(psi_des), np.sin(psi_des)])
        normal = np.array([-np.sin(psi_des), np.cos(psi_des)])
        vec = np.array([X - closest_node[0], Y - closest_node[1]])
        dy = np.dot(vec, normal)
        
        dpsi = wrap(psi - psi_des)

        vel = np.array([xdot, ydot])
        dy_dot = np.dot(vel, normal)
        dpsi_dot = psidot

        x_state = np.array([dy, dy_dot, dpsi, dpsi_dot])
        vx_design = max(xdot,0.1)
        K = self.design_lateral_controller_LQR(vx_design)

        #-- Adding feedforward section --#
        next2 = trajectory[(idx + 2) % len(trajectory)]
        dx1 = next_node[0] - closest_node[0]
        dy1 = next_node[1] - closest_node[1]
        dx2 = next2[0] - next_node[0]
        dy2 = next2[1] - next_node[1]

        seg1_len = np.hypot(dx1, dy1)
        seg2_len = np.hypot(dx2, dy2)
        if seg1_len > 1e-6 and seg2_len > 1e-6:
            ang1 = np.arctan2(dy1, dx1)
            ang2 = np.arctan2(dy2, dx2)
            dtheta = wrap(ang2 - ang1)
            curvature = dtheta / seg2_len
        else:
            curvature = 0.0

        wheelbase = self.lf + self.lr
        delta_ff = np.arctan(wheelbase * curvature)

        delta = -np.dot(K, x_state)
        # delta = clamp(u + delta_ff, -np.pi/6, np.pi/6)

        ## -- Longitudinal Controller --
        current_speed = np.sqrt(xdot**2 + ydot**2)
        F = self.longitudinal_pid.update(self.desired_speed, current_speed)
        # F = clamp(F, 0, 15736)

        return X, Y, xdot, ydot, psi, psidot, F, delta, obstacleX, obstacleY
