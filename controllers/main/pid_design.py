import numpy as np
from util import getTrajectory, closestNode, clamp

default_dt = 0.032

class PID:
    def __init__(self, Kp, Ki, Kd, dt=default_dt):
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

# Load trajectory using util.py
trajectory = getTrajectory('buggyTrace.csv')

# Simulate lateral controller (returns total squared cross-track error)
def lateral_error(pid_gains, trajectory, dt=default_dt):
    Kp, Ki, Kd = pid_gains
    pid = PID(Kp, Ki, Kd, dt)
    X, Y = trajectory[0, 0], trajectory[0, 1]  # start at first waypoint
    total_error = 0
    for i in range(1, len(trajectory)):
        cross_track_error, idx = closestNode(X, Y, trajectory)
        control = pid.update(0, cross_track_error)  # setpoint is 0
        # Simulate: move towards next waypoint, apply steering 
        direction = trajectory[i] - np.array([X, Y])
        speed = 11 #
        max_steer = np.radians(30)
        control = clamp(control, -max_steer, max_steer)
        X += direction[0] * dt + control * dt
        Y += direction[1] * dt + control * dt
        total_error += cross_track_error ** 2
    return total_error

# Simulate longitudinal controller (returns total squared speed error)
def longitudinal_error(pid_gains, trajectory, desired_speed=11.0, dt=default_dt):
    Kp, Ki, Kd = pid_gains
    pid = PID(Kp, Ki, Kd, dt)
    speed = 0.0
    total_error = 0
    for i in range(1, len(trajectory)):
        speed_error = desired_speed - speed
        control = pid.update(desired_speed, speed)
        # Simulate: update speed
        speed += control * dt
        speed = clamp(speed, 0, 50)  # clamp to reasonable speed range
        total_error += speed_error ** 2
    return total_error

# Twiddle algorithm for PID tuning
def twiddle(error_func, trajectory, tol=0.01, initial_p=[0.00, 0.0, 0.0], initial_dp=[0.01, 0.01, 0.01]):
    p = initial_p.copy()
    dp = initial_dp.copy()
    best_err = error_func(p, trajectory)
    it = 0
    prev_err = None
    while sum(dp) > tol:
        for i in range(len(p)):
            p[i] += dp[i]
            err = error_func(p, trajectory)
            if err < best_err:
                best_err = err
                dp[i] *= 1.1
            else:
                p[i] -= 2 * dp[i]
                err = error_func(p, trajectory)
                if err < best_err:
                    best_err = err
                    dp[i] *= 1.05
                else:
                    p[i] += dp[i]
                    dp[i] *= 0.95
        it += 1
        print(f"Iteration {it}, PID: {p}, Error: {best_err}")
        # Stop if error does not change
        if prev_err is not None and abs(best_err - prev_err) < 1e-8:
            print("Error did not change, stopping Twiddle.")
            break
        prev_err = best_err
    return p

if __name__ == "__main__":
    print("Tuning lateral PID...")
    best_pid_lateral = twiddle(lateral_error, trajectory)
    print("Best lateral PID gains:", best_pid_lateral)
    # dp's set to 0.1 - Best lateral PID gains: [0.1, 0.0, 0.0]
    # dp s set to 0.01 - Best lateral PID gains: [0.11435888100000004, 0.0, 0.0178966525]
    print("Tuning longitudinal PID...")
    best_pid_longitudinal = twiddle(longitudinal_error, trajectory)
    print("Best longitudinal PID gains:", best_pid_longitudinal)
    # dp's set to 0.1- Best longitudinal PID gains: [29.91268053287073, 1.3640345468017792, 0.02870992997411423]
    # dp's set to 0.01 -Best longitudinal PID gains: [27.580149049219923, 0.4822354299071474, 0.0734521331616762]