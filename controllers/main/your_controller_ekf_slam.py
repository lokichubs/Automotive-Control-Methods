# Fill in the respective function to implement the LQR/EKF SLAM controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from scipy.spatial.transform import Rotation
from util import *
from ekf_slam import EKF_SLAM


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

# CustomController class (inherits from BaseController)
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
        self.desired_speed = 26

        e1_max = 0.5
        e2_max = 8
        e3_max = 0.1
        e4_max = 1.4
        del_max = 0.1
        
        self.Q = np.diag([1/(e1_max**2), 1/(e2_max**2), 1/(e3_max**2), 1/(e4_max**2)])
        self.R = np.array([[1/(del_max**2)]])
        
        self.counter = 0
        np.random.seed(99)

        # Add additional member variables according to your need here.

    def getStates(self, timestep, use_slam=False):

        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Initialize the EKF SLAM estimation
        if self.counter == 0:
            # Load the map
            minX, maxX, minY, maxY = -120., 450., -500., 50.
            map_x = np.linspace(minX, maxX, 7)
            map_y = np.linspace(minY, maxY, 7)
            map_X, map_Y = np.meshgrid(map_x, map_y)
            map_X = map_X.reshape(-1,1)
            map_Y = map_Y.reshape(-1,1)
            self.map = np.hstack((map_X, map_Y)).reshape((-1))
            
            # Parameters for EKF SLAM
            self.n = int(len(self.map)/2)             
            X_est = X + 0.5
            Y_est = Y - 0.5
            psi_est = psi - 0.02
            mu_est = np.zeros(3+2*self.n)
            mu_est[0:3] = np.array([X_est, Y_est, psi_est])
            mu_est[3:] = np.array(self.map)
            init_P = 1*np.eye(3+2*self.n)
            W = np.zeros((3+2*self.n, 3+2*self.n))
            W[0:3, 0:3] = delT**2 * 0.1 * np.eye(3)
            V = 0.1*np.eye(2*self.n)
            V[self.n:, self.n:] = 0.01*np.eye(self.n)
            # V[self.n:] = 0.01
            print(V)
            
            # Create a SLAM
            self.slam = EKF_SLAM(mu_est, init_P, delT, W, V, self.n)
            self.counter += 1
        else:
            mu = np.zeros(3+2*self.n)
            mu[0:3] = np.array([X, 
                                Y, 
                                psi])
            mu[3:] = self.map
            y = self._compute_measurements(X, Y, psi)
            mu_est, _ = self.slam.predict_and_correct(y, self.previous_u)

        self.previous_u = np.array([xdot, ydot, psidot])

        print("True      X, Y, psi:", X, Y, psi)
        print("Estimated X, Y, psi:", mu_est[0], mu_est[1], mu_est[2])
        print("-------------------------------------------------------")
        
        if use_slam == True:
            return delT, mu_est[0], mu_est[1], xdot, ydot, mu_est[2], psidot
        else:
            return delT, X, Y, xdot, ydot, psi, psidot

    def _compute_measurements(self, X, Y, psi):
        x = np.zeros(3+2*self.n)
        x[0:3] = np.array([X, Y, psi])
        x[3:] = self.map
        
        p = x[0:2]
        psi = x[2]
        m = x[3:].reshape((-1,2))

        y = np.zeros(2*self.n)

        for i in range(self.n):
            y[i] = np.linalg.norm(m[i, :] - p)
            y[self.n+i] = wrapToPi(np.arctan2(m[i,1]-p[1], m[i,0]-p[0]) - psi)
            
        y = y + np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V)
        # print(np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V))
        return y
    
    def design_lateral_controller_LQR(self,v_design,delT):
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

        # Fetch the states from the newly defined getStates method
        delT, X, Y, xdot, ydot, psi, psidot = self.getStates(timestep, use_slam=True)
        # You must not use true_X, true_Y and true_psi since they are for plotting purpose
        _, true_X, true_Y, _, _, true_psi, _ = self.getStates(timestep, use_slam=False)

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
        
        
        dpsi = wrapToPi(psi - psi_des)

        vel = np.array([xdot, ydot])
        dy_dot = np.dot(vel, normal)
        dpsi_dot = psidot

        x_state = np.array([dy, dy_dot, dpsi, dpsi_dot])
        vx_design = max(xdot,0.1)
        K = self.design_lateral_controller_LQR(vx_design,delT)

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
            dtheta = wrapToPi(ang2 - ang1)
            curvature = dtheta / seg2_len
        else:
            curvature = 0.0

        wheelbase = self.lf + self.lr
        delta_ff = np.arctan(wheelbase * curvature)

        delta = -np.dot(K, x_state)

        ## -- Longitudinal Controller --
        current_speed = np.sqrt(xdot**2 + ydot**2)
        F = self.longitudinal_pid.update(self.desired_speed, current_speed)

        # Return all states and calculated control inputs (F, delta)
        return true_X, true_Y, xdot, ydot, true_psi, psidot, F, delta
