# (Replace your existing controller file with this)
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

        # --- Vehicle constants ---
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

        # --- Controller ---
        self.delT = 0.032
        self.longitudinal_pid = PID(Kp=16.58, Ki=0.482, Kd=0.0734, dt=self.delT)
        self.desired_speed = 20

        # --- Lateral gain ---
        self.K = self.design_lateral_controller()
        print("Lateral gain K:", self.K)

    def design_lateral_controller(self):
        """Calculates the state-feedback gain vector K using discrete pole placement."""

        lr, lf, Ca, Iz, m, v_design, delT = self.lr, self.lf, self.Ca, self.Iz, self.m, self.desired_speed, self.delT

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
        continuous_poles = np.array([-4, -100, -150, -125])
        desired_poles_z = np.exp(continuous_poles * delT)

        try:
            result = signal.place_poles(Ad, Bd, desired_poles_z, method='YT')
            K = result.gain_matrix
            return np.asarray(K).reshape(-1)
        except Exception as e:
            print(f"Discrete pole placement failed: {e}. Returning zero gain.")
            return np.zeros(4)

    def update(self, timestep):
        trajectory = self.trajectory

        # Fetch states
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)
        self.longitudinal_pid.dt = delT  

        # ---------------- Lateral Controller ----------------
        cross_track_error, idx = closestNode(X, Y, trajectory)
        closest_node = trajectory[idx]
        next_node = trajectory[(idx + 1) % len(trajectory)]

        psi_des = np.arctan2(next_node[1] - closest_node[1], next_node[0] - closest_node[0])

        # Signed cross-track error
        t_dx = next_node[0] - closest_node[0]
        t_dy = next_node[1] - closest_node[1]
        v_dx = X - closest_node[0]
        v_dy = Y - closest_node[1]
        signed_cte = np.sign(t_dx * v_dy - t_dy * v_dx) * cross_track_error

        dy = signed_cte
        dpsi = wrap(psi_des - psi)
        dy_dot = ydot
        dpsi_dot = psidot

        x_state = np.array([dy, dy_dot, dpsi, dpsi_dot])
        if abs(dy) < 0.05:
            raw_u = 0 

        raw_u = -np.dot(self.K*1.8e-5, x_state)
        delta = clamp(raw_u, -np.pi/6, np.pi/6)

        # ---------------- Longitudinal Controller ----------------
        current_speed = np.sqrt(xdot**2 + ydot**2)
        F = self.longitudinal_pid.update(self.desired_speed, current_speed)
        F = clamp(F, 0, 15736)

        # print("Delta (deg):", np.degrees(delta), "Force (N):", F)
        return X, Y, xdot, ydot, psi, psidot, F, delta
