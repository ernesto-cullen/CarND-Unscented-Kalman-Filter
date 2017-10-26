#include "ukf.h"
#include "Eigen/Dense"
#include "tools.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;
  n_x_aug_ = 7;
  n_sig_ = 2*n_x_+1;
  n_sig_aug_ = 2*n_x_aug_+1;
  lambda_ = 3-n_x_;

  // initial state vector
  x_ = VectorXd(n_x_);
  // augmented state vector
  x_aug_ = VectorXd(n_x_aug_);
  // predicted sigma points
  Xsig_pred_ = MatrixXd(n_x_, n_sig_aug_);
  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 6;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.0175;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.1;

  // compute weights
  weights_ = VectorXd(n_sig_aug_);
  weights_.fill(0.5*1/(lambda_+n_x_aug_));
  weights_(0) = lambda_/(lambda_+n_x_aug_);

  // Measurement noise covariance matrices
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
      0, std_radphi_*std_radphi_, 0,
      0, 0,std_radrd_*std_radrd_;

  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_*std_laspx_,0,
      0,std_laspy_*std_laspy_;

  is_initialized_ = false;
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

/// Generate a matrix with sigma points without noise information
/// \param Xsig_out matrix to put the sigma points in, one per column
///
void UKF::GenerateSigmaPoints(MatrixXd *Xsig) {
  MatrixXd Xsig_out = MatrixXd(n_x_, n_sig_);
  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();
  MatrixXd sqrtA = sqrt(lambda_+n_x_)*A;
  //first sigma point is the mean vector
  Xsig_out.col(0) = x_;
  //calculate next 2*n_x_ sigma points
  for (int i=0; i<n_x_; i++) {
    Xsig_out.col(1+i) = x_+sqrtA.col(i);
    Xsig_out.col(1+n_x_+i) = x_-sqrtA.col(i);
  }
  //copy results to Xsig_out
  *Xsig = Xsig_out;
}

/// Generate sigma points matrix considering process noise
/// \param Xsig_out
void UKF::GenerateAugmentedSigmaPoints(MatrixXd *Xsig_aug) {
  MatrixXd Xsig = MatrixXd(n_x_aug_, n_sig_aug_);

  //augmented mean state
  x_aug_.fill(0);
  x_aug_.head(n_x_) = x_;

  MatrixXd Q = MatrixXd(2,2);
  Q << std_a_*std_a_, 0,
      0, std_yawdd_*std_yawdd_;

  // augmented P
  MatrixXd P_aug = MatrixXd(n_x_aug_, n_x_aug_);
  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug.bottomRightCorner(2,2) = Q;

  // square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  MatrixXd sqrtA = sqrt(lambda_+n_x_aug_)*A;

  // augmented sigma points
  Xsig.col(0) = x_aug_;
  for (int i=0; i<n_x_aug_; i++) {
    Xsig.col(1+i) = x_aug_ + sqrtA.col(i);
    Xsig.col(1+n_x_aug_+i) = x_aug_ - sqrtA.col(i);
  }
  //copy results to Xsig_out
  *Xsig_aug = Xsig;
}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /*****************************************************************************
  *  Initialization
  ****************************************************************************/
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

  //elapsed time in seconds
  double dt = (meas_package.timestamp_ - previous_timestamp_) / 1.e6;

  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else {
    UpdateLidar(meas_package);
  }

  previous_timestamp_ = meas_package.timestamp_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // sigma points (including process noise)
  MatrixXd Xsig_aug = MatrixXd(n_x_aug_, n_sig_aug_);
  GenerateAugmentedSigmaPoints(&Xsig_aug);

  // predicted sigma points

  PredictSigmaPoints(delta_t, &Xsig_aug);

  // calculate predicted mean and covariance using predicted sigma points
  PredictMeanAndCovariance();

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int n_z = 2;
  // sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_aug_);

  //transform sigma points into measurement space
  for (int i=0; i<n_sig_aug_; i++) {
    Zsig.col(i) = Xsig_pred_.col(i).head(2);
  }
  Update(meas_package, Zsig);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  int n_z = 3;
  // sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_aug_);

  //transform sigma points into measurement space
  for (int i = 0; i < n_sig_aug_; i++) {
    VectorXd x = Xsig_pred_.col(i);
    double rho = sqrt(x(0) * x(0) + x(1) * x(1));
    double phi = atan2(x(1), x(0));
    double phip = (x(0) * cos(x(3)) * x(2) + x(1) * sin(x(3)) * x(2)) / rho;
    Zsig.col(i) << rho, phi, phip;
  }

  Update(meas_package, Zsig);
}

void UKF::Update(MeasurementPackage meas_package, MatrixXd Zsig) {
  MatrixXd R;
  int n_z;

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    n_z = 3;
    R = R_radar_;
  } else {
    n_z = 2;
    R = R_lidar_;
  }

  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);
  for (int i=0; i<n_sig_aug_; i++) {
    z_pred += weights_(i)*Zsig.col(i);
  }

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0);
  for(int i=0; i<n_sig_aug_; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i)-z_pred;
    //angle normalization
    while (z_diff(1)>M_PI) z_diff(1) -= M_PI;
    while (z_diff(1)<-M_PI) z_diff(1) += M_PI;
    S += weights_(i)*z_diff*z_diff.transpose();
  }

  S += R;

  // cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0);
  for (int i=0; i<n_sig_aug_; i++) {
    VectorXd z_diff = Zsig.col(i)-z_pred;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      //angle correction
      while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
      while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
    }

    VectorXd x_diff = Xsig_pred_.col(i)-x_;
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc += weights_(i)*x_diff*z_diff.transpose();
  }

  // Kalman gain
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd z = meas_package.raw_measurements_;
  x_ = x_ + K * (z - z_pred);
  P_ = P_ - K * S * K.transpose();

  // NIS
  double nis = z.transpose() * S.inverse() * z;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
    NIS_radar_ = nis;
  }
  else {
    NIS_laser_ = nis;
  }

}

MatrixXd UKF::PredictSigmaPoints(double dt, MatrixXd *Xsig_aug) {
  VectorXd v1 = VectorXd(n_x_);
  VectorXd v2 = VectorXd(n_x_);

  for (int i=0; i<n_sig_aug_; i++) {
    VectorXd x_k = Xsig_aug -> col(i);
    double v_k = x_k(2);
    double yaw_k = x_k(3);
    double yawp_k = x_k(4);
    double nu_ak = x_k(5);
    double nu_yawppk = x_k(6);

    v2 << 0.5 * dt * dt * cos(yaw_k) * nu_ak,
        0.5 * dt * dt * sin(yaw_k) * nu_ak,
        dt * nu_ak,
        0.5 * dt * dt * nu_yawppk,
        dt * nu_yawppk;

    if (abs(yawp_k > 0.00001)) {
      v1 << v_k / yawp_k * (sin(yaw_k + yawp_k * dt) - sin(yaw_k)),
          v_k / yawp_k * (-cos(yaw_k + yawp_k * dt) + cos(yaw_k)),
          0,
          yawp_k * dt,
          0;
    } else {
      v1 << v_k * cos(yaw_k) * dt,
          v_k * sin(yaw_k) * dt,
          0,
          0,
          0;
    }
    Xsig_pred_.col(i) = x_k.head(n_x_) + v1 + v2;
  }
}

void UKF::PredictMeanAndCovariance() {
  // compute mean
  x_.fill(0);
  for (int i=0; i<n_sig_aug_; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // compute state covariance matrix
  P_.fill(0);
  for (int i=0; i<n_sig_aug_; i++) {
    VectorXd diff = (Xsig_pred_.col(i) - x_);
    //angle normalization
    while (diff(3)> M_PI) diff(3)-=2.*M_PI;
    while (diff(3)<-M_PI) diff(3)+=2.*M_PI;

    diff(3) = atan2(sin(diff(3)),cos(diff(3)));
    P_ += weights_(i)*diff*diff.transpose();
  }
}



