# Automotive-Control-Methods

## PID

### Method
Implemented two PID controllers for lateral and longitudinal controls respectively.Gains were tuned using the [twiddle algorithm](controllers/main/pid_design.py). A custom signed cross-track error was used in the lateral control and a differential error was used for the longitudinal controller.

### Results
Trajectory following and control performance are shown below: (find all attempts [here](assets/img/PID))

![PID Results](assets/img/PID/Attempt_9.png)

Simulation video is shown below:
<video src="assets/vid/PID/pid_sim_vid.mp4" controls width="600"></video>
