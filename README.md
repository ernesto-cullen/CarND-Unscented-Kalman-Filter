# Unscented Kalman Filter Project
Self-Driving Car Engineer Nanodegree Program


In this project we write an Unscented Kalman Filter to estimate the state of a moving object with noisy lidar and radar measurements. Measurements from these two sensors will be fused to improve the estimation.


## Dependencies and environment setup

This project uses the [Term 2 Simulator](https://github.com/udacity/self-driving-car-sim/releases). The simulator will show a car doing a predetermined circuit, along with the measurements taken and the estimations computed. The simulator communicates with the project using uWebSockets library to get the estimations as they are calculated, as long as the root mean squared error (RMSE) against the real data (ground thruth).

I had a hard time setting up the environment. I work on Windows 10, but getting uWebSockets to run is difficult on this OS so I tried on a Ubuntu Virtual Machine. I had used this VM in previous EKF project so the uWebSockets library and the simulator were already installed and working. BUT it seems since the EKF project something has changed, either in Linux or in the VM: the simulator won't start anymore. I searched the forums (a lot of people had this problem too), the general web and try different versions of the simulator. Nothing worked.
Then I tried installing the simulator on Windows (where it runs perfectly) and the code in Bash for Windows 10, which is the setup recommended in Udacity's project instructions. It worked at first, but after some days (I think it was after Win 10 Creator's Update installation) the simulator stopped to communicate with the program running on Bash for Windows.
Tried also using the simulator on Windows and the program in the VM using [this trick](https://discussions.udacity.com/t/running-simulator-on-windows-and-code-on-ubuntu/255869). Again, this worked at first and after some days it stopped being able to commuicate with the simulator.
Finally, I had to install a 'real' ubuntu 16.04 in a computer in order to get the simulator working. Using [CLion IDE](https://www.jetbrains.com/clion/) I was able to run the code from inside the IDE while the simulator was running, which made the testing and debugging go much faster and easier. There seems to be some quirks in CLion's internal evaluator as we see some errors in the editor which are not real: we can compile and run without problems.


### Other Important Dependencies
* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

### Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./UnscentedKF`. You'll see a message saying the program is ready listening at port 4567
5. Start the simulator. Once the simulation starts, you'll see 'Connected!' in the console.



## The program

The program will simulate the real process:

1. take an initial measurement, use it to initialize the system
2. predict the next state of the vehicle
3. get a new measurement, adjust the prediction
4. repeat from point 2.

In this case, the project use a file with entries simulating the sensor measurements, interspersing laser and radar measurements.


### Initialization
The first measurement will be used to initialize the system. Code is found in UKF::ProcessMeasurement method, which takes as input the sensor data for one measure:

```C++
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  //initialize
  if (!is_initialized_) {
    previous_timestamp_ = meas_package.timestamp_;
    P_ = MatrixXd::Identity(n_x_, n_x_);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      x_ = Tools::polar2cartesian(meas_package.raw_measurements_);
    }
    else {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0., 0., 0.;
    }

    is_initialized_ = true;
    return;
  }
```

the P matrix (covariance matrix) is initialized as the identity matrix, then the state variable is set according to the type of sensor the measurement comes from.

The project uses the CTRV (Constant Turn Rate and Velocity Magnitude) model. The state in this model is a vector of 5 components:

![CTRV state vector](images/state_vector.png)

The magnitudes can be seen in the following graph:

![CTRV state vector](images/ctrv_model.png)

if the initial measurement comes from radar, it will contain the polar coordinates of the vehicule; we have to convert to cartesian coordinates to get the state. This is done in utility class Tools:

 ```C++
 VectorXd Tools::polar2cartesian(VectorXd pos) {
   VectorXd result(5);
   double rho = pos(0);
   double phi = pos(1);
   double phi_dot = pos(2);

   double px = rho * cos(phi);
   double py = rho * sin(phi);
   double vx = phi_dot * cos(phi);
   double vy = phi_dot * sin(phi);

   result << px, py, sqrt(vx*vx+vy*vy), 0, 0;
   return result;
 }
```

we can set the position and linear velocity of the object; the yaw angle and its rate are initialized to 0.
In the case of a Lidar measurement, it only has the cartesian coordinates of the object; again, the missing elements are initialized to 0.

after the first measurement which initializes the state, the next measurement will trigger the Kalman Filter process: predict and update. We need the time elapsed in between two cycles for this. Code is in the same `UKF::ProcessMeasurement` method:

```C++
  //elapsed time in seconds
  double dt = (meas_package.timestamp_ - previous_timestamp_) / 1.e6;

  //Predict
  Prediction(dt);

  //Update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }

  previous_timestamp_ = meas_package.timestamp_;
```

### Prediction
The first step of the Kalman cycle is the prediction of the position based on the previous one. In Unscented Kalman Filter, this prediction is based on _sigma points_ to deal with both linear and non linear situations.
The sigma points are points that are taken in a vicinity of the actual position, with a certain relation to the standard deviation sigma of every state dimension (that's were their name is derived). These points serve as a representation of the current state.
The sigma points are then transformed to the _predicted state space_, where you can find the mean and the standard deviation of a state that corresponds to these transformed sigma points. This new state will be the _predicted state_.

The prediction step is implemented in `UKF::Prediction` method, called from the same `ProcessMeasurement` as before:

```C++
void UKF::Prediction(double delta_t) {
  // sigma points (including process noise)
  GenerateAugmentedSigmaPoints();

  // predicted sigma points
  PredictSigmaPoints(delta_t);

  // calculate predicted mean and covariance using predicted sigma points
  PredictMeanAndCovariance();
}
```

As a result of this step, we will get a _predicted state_ and its corresponding deviation.

### Update
After the prediction, the state is updated with the real measurement. This will bring the estimation much closer to the real value (which is not exactly the measured value because the measurement has noise). As the measurement comes from different sensors, each has to be processed separately

```C++
  //Update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }
```

### Error measurement
After this, the state is sent to the simulator for grafication, while also computing the deviation from the _ground truth_ data which is provided. This deviation is represented by the RMSE (Root Mean Squared Error) which is computed also in Tools class:

```C++
VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse.fill(0);

  for (int k = 0; k < estimations.size(); ++k){
    VectorXd diff = estimations[k] - ground_truth[k];
    diff = diff.array() * diff.array();
    rmse += diff;
  }

  rmse /= (double)estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}
```

This computation is exactly the same as in the EKF project.


## Conclusion
The CTRV model is able to follow very closely the trajectory of the object, with a smooth curve joining the successive states. It is specially well suited for objects that follow linear or circular paths like the car in the simulation.

The UKF allows for the processing of nonlinearities without using linearization like in EKF; just using simple point transformations. For a detailed look at the mathematics, see the course material or other resources like the paper [The Unscented Kalman Filter for Nonlinear Estimation](https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf) or [this entry on Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter). The implementation here is based on the course lesson with some small optimizations in the code.

