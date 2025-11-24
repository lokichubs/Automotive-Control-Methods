import numpy as np

class EKF_SLAM():
    def __init__(self, init_mu, init_P, dt, W, V, n):
        r"""Initialize EKF SLAM

        Create and initialize an EKF SLAM to estimate the robot's pose and
        the location of map features

        Args:
            init_mu: A numpy array of size (3+2*n, ). Initial guess of the mean 
            of state. 
            init_P: A numpy array of size (3+2*n, 3+2*n). Initial guess of 
            the covariance of state.
            dt: A double. The time step.
            W: A numpy array of size (3+2*n, 3+2*n). Process noise
            V: A numpy array of size (2*n, 2*n). Observation noise
            n: A int. Number of map features
            

        Returns:
            An EKF SLAM object.
        """
        self.mu = init_mu  # initial guess of state mean
        self.P = init_P  # initial guess of state covariance
        self.dt = dt  # time step
        self.W = W  # process noise 
        self.V = V  # observation noise
        self.n = n  # number of map features

    # TODO: complete the function below
    def _f(self, x, u):
        r"""Non-linear dynamic function.

        Compute the state at next time step according to the nonlinear dynamics f.

        Args:
            x: A numpy array of size (3+2*n, ). State at current time step.
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            x_next: A numpy array of size (3+2*n, ). The state at next time step
        """
        X, Y, psi = x[0], x[1], x[2]
        dx,dy,dpsi = u

        x_next = np.copy(x)
        x_next[0] = X + self.dt*(u[0]*np.cos(psi) - u[1]*np.sin(psi))
        x_next[1] = Y + self.dt*(u[0]*np.sin(psi) + u[1]*np.cos(psi))
        x_next[2] = psi + self.dt*u[2]

        return x_next

    # TODO: complete the function below
    def _h(self, x):
        r"""Non-linear measurement function.

        Compute the sensor measurement according to the nonlinear function h.

        Args:
            x: A numpy array of size (3+2*n, ). State at current time step.

        Returns:
            y: A numpy array of size (2*n, ). The sensor measurement.
        """
        X, Y, psi = x[0], x[1], x[2]
        y = np.zeros(2*self.n)

        for j in range(self.n):
            mx = x[3+2*j]
            my = x[3+2*j + 1]

            dx = mx - X
            dy = my - Y

            r = np.sqrt(dx**2 + dy**2)
            b = np.arctan2(dy,dx) - psi

            y[j] = r
            y[self.n+j] = b


        return y

    # TODO: complete the function below
    def _compute_F(self, x, u):
        r"""Compute Jacobian of f

        Args:
            x: A numpy array of size (3+2*n, ). The state vector.
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            F: A numpy array of size (3+2*n, 3+2*n). The jacobian of f evaluated at x_k.
        """
        X, Y, psi = x[0], x[1], x[2]
        dx, dy, dpsi = u

        F_11 =  np.eye(3)
        F_11[0,2] = self.dt*(-dx*np.sin(psi) - dy*np.cos(psi))
        F_11[1,2] = self.dt*( dx*np.cos(psi) - dy*np.sin(psi))

        F_12 = np.zeros((3,2*self.n))
        F_21 = np.zeros((2*self.n,3))
        F_22 = np.eye(2*self.n)

        F = np.block([
            [F_11, F_12],
            [F_21, F_22]
        ])

        return F

    # TODO: complete the function below
    def _compute_H(self, x):
        r"""Compute Jacobian of h

        Args:
            x: A numpy array of size (3+2*n, ). The state vector.

        Returns:
            H: A numpy array of size (2*n, 3+2*n). The jacobian of h evaluated at x_k.
        """
        X, Y, psi = x[0], x[1], x[2]
        n = self.n

        H = np.zeros((2*n, 3 + 2*n))

        for i in range(n):
            mx = x[3+2*i]
            my = x[3+2*i + 1] 

            dx = mx - X
            dy = my - Y 

            q = dx**2 + dy**2
            r = np.sqrt(q)

            # Distance rows
            H[i,0] = -dx/r
            H[i,1] = -dy/r
            H[i,2] = 0

            H[i,3+2*i] = dx/r
            H[i,3+2*i+1] = dy/r

            # Bearing rows
            H[n+i,0] = dy/q
            H[n+i,1] = -dx/q
            H[n+i,2] = -1

            H[n+i,3+2*i] = -dy/q
            H[n+i,3+2*i+1] = dx/q



        return H


    def predict_and_correct(self, y, u):
        r"""Predice and correct step of EKF

        Args:
            y: A numpy array of size (2*n, ). The measurements according to the project description.
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            self.mu: A numpy array of size (3+2*n, ). The corrected state estimation
            self.P: A numpy array of size (3+2*n, 3+2*n). The corrected state covariance
        """

        # compute F
        F = self._compute_F(self.mu, u)

        #***************** Predict step *****************#
        # predict the state
        self.mu = self._f(self.mu, u)
        self.mu[2] =  self._wrap_to_pi(self.mu[2])
        # predict the error covariance
        self.P = F @ self.P @ F.T + self.W

        #***************** Correct step *****************#
        # compute H matrix
        H = self._compute_H(self.mu)

        # compute the Kalman gain
        L = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.V)

        # update estimation with new measurement
        diff = y - self._h(self.mu)
        diff[self.n:] = self._wrap_to_pi(diff[self.n:])
        self.mu = self.mu + L @ diff
        self.mu[2] =  self._wrap_to_pi(self.mu[2])

        # update the error covariance
        self.P = (np.eye(3+2*self.n) - L @ H) @ self.P

        return self.mu, self.P


    def _wrap_to_pi(self, angle):
        angle = angle - 2*np.pi*np.floor((angle+np.pi )/(2*np.pi))
        return angle


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    m = np.array([[0.,  0.],
                  [0.,  20.],
                  [20., 0.],
                  [20., 20.],
                  [0,  -20],
                  [-20, 0],
                  [-20, -20],
                  [-50, -50]]).reshape(-1)

    dt = 0.01
    T = np.arange(0, 20, dt)
    n = int(len(m)/2)
    W = np.zeros((3+2*n, 3+2*n))
    W[0:3, 0:3] = dt**2 * 1 * np.eye(3)
    V = 0.1*np.eye(2*n)
    V[n:,n:] = 0.01*np.eye(n)

    # EKF estimation
    mu_ekf = np.zeros((3+2*n, len(T)))
    mu_ekf[0:3,0] = np.array([2.2, 1.8, 0.])
    # mu_ekf[3:,0] = m + 0.1
    mu_ekf[3:,0] = m + np.random.multivariate_normal(np.zeros(2*n), 0.5*np.eye(2*n))
    init_P = 1*np.eye(3+2*n)

    # initialize EKF SLAM
    slam = EKF_SLAM(mu_ekf[:,0], init_P, dt, W, V, n)
    
    # real state
    mu = np.zeros((3+2*n, len(T)))
    mu[0:3,0] = np.array([2, 2, 0.])
    mu[3:,0] = m

    y_hist = np.zeros((2*n, len(T)))
    for i, t in enumerate(T):
        if i > 0:
            # real dynamics
            u = [-5, 2*np.sin(t*0.5), 1*np.sin(t*3)]
            # u = [0.5, 0.5*np.sin(t*0.5), 0]
            # u = [0.5, 0.5, 0]
            mu[:,i] = slam._f(mu[:,i-1], u) + \
                np.random.multivariate_normal(np.zeros(3+2*n), W)

            # measurements
            y = slam._h(mu[:,i]) + np.random.multivariate_normal(np.zeros(2*n), V)
            y_hist[:,i] = (y-slam._h(slam.mu))
            # apply EKF SLAM
            mu_est, _ = slam.predict_and_correct(y, u)
            mu_ekf[:,i] = mu_est


    plt.figure(1, figsize=(10,6))

    # Trajectory
    ax1 = plt.subplot(121, aspect='equal')
    ax1.plot(mu[0,:], mu[1,:], 'b', label='GT Trajectory')
    ax1.plot(mu_ekf[0,:], mu_ekf[1,:], 'r--', label='EKF Trajectory')
    mf = m.reshape((-1,2))
    ax1.scatter(mf[:,0], mf[:,1], label='Landmarks')
    ax1.legend()
    ax1.set_title("Robot Trajectory and Landmark Map")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # X
    ax2 = plt.subplot(322)
    ax2.plot(T, mu[0,:], 'b', label="GT X")
    ax2.plot(T, mu_ekf[0,:], 'r--', label="EKF X")
    ax2.legend()
    ax2.set_xlabel('t')
    ax2.set_ylabel('X')

    # Y
    ax3 = plt.subplot(324)
    ax3.plot(T, mu[1,:], 'b', label="GT Y")
    ax3.plot(T, mu_ekf[1,:], 'r--', label="EKF Y")
    ax3.legend()
    ax3.set_xlabel('t')
    ax3.set_ylabel('Y')

    # psi
    ax4 = plt.subplot(326)
    ax4.plot(T, mu[2,:], 'b', label="GT psi")
    ax4.plot(T, mu_ekf[2,:], 'r--', label="EKF psi")
    ax4.legend()
    ax4.set_xlabel('t')
    ax4.set_ylabel('psi')

    # Figure 2
    plt.figure(2)
    ax5 = plt.subplot(211)
    ax5.plot(T, y_hist[0:n, :].T)
    ax5.set_title("Range Measurement Residuals")

    ax6 = plt.subplot(212)
    ax6.plot(T, y_hist[n:, :].T)
    ax6.set_title("Bearing Measurement Residuals")

    plt.tight_layout()
    plt.show()
