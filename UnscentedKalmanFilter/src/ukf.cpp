#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // Init bool to false
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Choose state dimensions
  n_x_ = 5;
  n_aug_ = 7;
  n_z_radar_ = 3;
  n_z_laser_ = 2;

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Zero(n_x_, n_x_);

  // init predicted sigma points matrix
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.25 * M_PI;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  // Choose lambda
  lambda_ = 9 - n_aug_;
  
  // Init weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  // Set weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }
  // std::cout << "Weights " << std::endl << weights_ << std::endl << std::endl;

  // Choose measurement covariance matrix for radar
  R_radar_ = MatrixXd(n_z_radar_,n_z_radar_);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
          0, std_radphi_ * std_radphi_, 0,
          0, 0,std_radrd_ * std_radrd_;

  R_laser_ = MatrixXd(n_z_laser_,n_z_laser_);
  R_laser_ << std_laspx_ * std_laspx_, 0,
          0, std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  // Initialize
  if(!is_initialized_){
    // Save time stamp
    time_us_ = meas_package.timestamp_;

    // Init state vector
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      // Convert Polar coordinates into Cartesian coordinates
      x_(0) = meas_package.raw_measurements_[0] 
        * cos(meas_package.raw_measurements_[1]);
      x_(1) = meas_package.raw_measurements_[0] 
        * sin(meas_package.raw_measurements_[1]);
    }
    else{
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
    }

    // Init state covariance
    P_ <<  1,  0,  0,  0,  0,
           0,  1,  0,  0,  0,
           0,  0, 10,  0,  0,
           0,  0,  0,  1,  0,
           0,  0,  0,  0,  1;

    // Set init flag to true
    is_initialized_ = true;

    // Print status
    // std::cout << "Initialization" << std::endl;
    // std::cout << "x = " << std::endl << x_ << std::endl;
    // std::cout << "P = " << std::endl << P_ << std::endl;
    // std::cout << std::endl << std::endl;
  }
  // Process measurement
  else{
    // Calculate delta t and save current time stamp
    float delta_t = (meas_package.timestamp_ - time_us_)
       / 1000000.0;
    time_us_ = meas_package.timestamp_;

    // Prediction
    Prediction(delta_t);

    // Update
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
      UpdateRadar(meas_package);
    }
    else if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
      UpdateLidar(meas_package);
    }

  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  // std::cout << "Predict with dt = " << delta_t << " s" << std::endl;

  // 1. Generate augmented sigma points
  //create augmented mean vector and state covariance
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++){
    Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  // std::cout << "Generated sigma points " << std::endl << Xsig_aug << std::endl << std::endl;

  // 2. Predict sigma points
  for (int i = 0; i< 2 * n_aug_ + 1; i++){
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd * delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw + yawd * delta_t) );
    }
    else {
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  // std::cout << "Predicted sigma points " << std::endl << Xsig_pred_ << std::endl << std::endl;

  // 3. Predict mean and covariance
  // Predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }

  // Print prediction
  std::cout << "Prediction" << std::endl;
  std::cout << "x = " << std::endl << x_ << std::endl;
  std::cout << "P = " << std::endl << P_ << std::endl;
  // std::cout << std::endl << std::endl;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  // std::cout << "Lidar Update with = " << std::endl 
  //   << meas_package.raw_measurements_ << std::endl;

  // 1. Predict measurement
  // Init measurement sigma points
  MatrixXd Zsig = Xsig_pred_.topLeftCorner(n_z_laser_,2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_laser_);
  z_pred.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z_laser_, n_z_laser_);
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_laser_);

  S.fill(0.0);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_sig_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_sig_diff(1) >  M_PI) z_sig_diff(1) -= 2. * M_PI;
    while (z_sig_diff(1) < -M_PI) z_sig_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_sig_diff * z_sig_diff.transpose();

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_sig_diff.transpose();
  }

  //add measurement noise covariance matrix
  S = S + R_laser_;

  // 2. State update

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  // Print
  // cout << "Laser Measurement" << endl;
  // std::cout << "z_pred " << std::endl << z_pred << std::endl;
  // std::cout << "z " << std::endl << meas_package.raw_measurements_ << std::endl;
  // std::cout << "z_diff " << std::endl << z_diff << std::endl;
  // std::cout << "S " << std::endl << S << std::endl;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  
    // Print prediction
  std::cout << "Laser Update" << std::endl;
  std::cout << "x = " << std::endl << x_ << std::endl;
  std::cout << "P = " << std::endl << P_ << std::endl;
  std::cout << "eps = " << std::endl << CalculateNIS(z_diff,S) << std::endl;
  // std::cout << std::endl << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  // std::cout << "Radar Update with = " << std::endl 
    // << meas_package.raw_measurements_ << std::endl;

  // 1. Predict measurement
  // Init measurement sigma points
  MatrixXd Zsig = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);

  // Predict measurement
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_radar_);
  z_pred.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);

  S.fill(0.0);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_sig_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_sig_diff(1) >  M_PI) z_sig_diff(1) -= 2. * M_PI;
    while (z_sig_diff(1) < -M_PI) z_sig_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_sig_diff * z_sig_diff.transpose();

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_sig_diff.transpose();
  }

  //add measurement noise covariance matrix
  S = S + R_radar_;

  // 2. State update

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // Print
  // cout << "Radar Measurement" << endl;
  // std::cout << "z_pred " << std::endl << z_pred << std::endl;
  // std::cout << "z " << std::endl << meas_package.raw_measurements_ << std::endl;
  // std::cout << "z_diff " << std::endl << z_diff << std::endl;
  // std::cout << "S " << std::endl << S << std::endl;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

    // Print prediction
  std::cout << "Radar Update" << std::endl;
  std::cout << "x = " << std::endl << x_ << std::endl;
  std::cout << "P = " << std::endl << P_ << std::endl;
  std::cout << "eps = " << std::endl << CalculateNIS(z_diff,S) << std::endl;
  // std::cout << std::endl << std::endl;
}

float UKF::CalculateNIS(const VectorXd z_diff, const MatrixXd S){
  
  return z_diff.transpose() * S.inverse() * z_diff;
}
