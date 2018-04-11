#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(const VectorXd &x_in) {

  // Set dimensions
  number_of_states_ = x_in.size();
  number_of_laser_observations_ = 2;
  number_of_radar_observations_ = 3;

  // Store initial state vector and covariance
  x_ = x_in;
  P_ = MatrixXd::Zero(number_of_states_,number_of_states_);
  P_ << 1000, 0, 0, 0,
        0, 1000, 0, 0,
        0, 0,   10, 0,
        0, 0,    0,10;

  // Set other matrices
  F_ = MatrixXd::Identity(number_of_states_,number_of_states_);
  H_ = MatrixXd::Identity(number_of_laser_observations_,number_of_states_);
  Q_ = MatrixXd::Zero(number_of_states_,number_of_states_);
  R_laser_ = MatrixXd::Zero(number_of_laser_observations_,number_of_laser_observations_);
  R_laser_ << 0.0225, 0,
        0, 0.0225;
  R_radar_ = MatrixXd::Zero(number_of_radar_observations_,number_of_radar_observations_);
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;
  I_ = MatrixXd::Identity(number_of_states_, number_of_states_);

  // Construct tool object
  tools = Tools();


  //set process noise
  noise_ax_ = 9.0;
  noise_ay_ = 9.0;

  // Print initial matrixes
  // cout << "INIT" << endl;
  // cout << "x_ " << endl << x_ << endl;
  // cout << "P_ " << endl << P_ << endl;
  // cout << "F_ " << endl << F_ << endl;
  // cout << "H_ " << endl << H_ << endl;
  // cout << "Q_ " << endl << Q_ << endl;
  // cout << "R_laser " << endl << R_laser_ << endl;
}

void KalmanFilter::Predict(const float delta_T) {

  //update state prediction matrix F
  F_(0,2) = delta_T;
  F_(1,3) = delta_T;
  //cout << "F_ " << endl << F_ << endl;

  //update process covariance matrix Q
  Q_(0,0) = std::pow(delta_T,4) / 4 * noise_ax_;
  Q_(0,2) = std::pow(delta_T,3) / 2 * noise_ax_;
  Q_(1,1) = std::pow(delta_T,4) / 4 * noise_ay_;
  Q_(1,3) = std::pow(delta_T,3) / 2 * noise_ay_;
  Q_(2,0) = std::pow(delta_T,3) / 2 * noise_ax_;
  Q_(2,2) = std::pow(delta_T,2) * noise_ax_;
  Q_(3,1) = std::pow(delta_T,3) / 2 * noise_ay_;
  Q_(3,3) = std::pow(delta_T,2) * noise_ay_;
  //cout << "Q_ " << endl << Q_ << endl;

  //predict state vector
  x_ = F_ * x_;

  //predict state covariance
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const Eigen::VectorXd &z){

  VectorXd z_pred = H_ * x_;
  //cout << "z_pred " << endl << z_pred << endl;
  //cout << "z_ " << endl << z << endl;
  VectorXd y = z - z_pred;
  //cout << "y_ " << endl << y << endl;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_laser_;
  //cout << "S_ " << endl << S << endl;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;
  //cout << "K_ " << endl << K << endl;

  //new estimate
  x_ = x_ + (K * y);   
  P_ = (I_ - K * H_) * P_;

  // print predicted state
  //cout << "x_updL " << endl << x_ << endl;
  //cout << "P_updL " << endl << P_ << endl;
}

void KalmanFilter::UpdateEKF(const Eigen::VectorXd &z){


  MatrixXd JH = tools.CalculateJacobian(x_);
  //cout << "JH " << endl << JH << endl;

  VectorXd z_pred = tools.ConvertCartesianIntoPolar(x_);
  // cout << "z_pred " << endl << z_pred << endl;
  // cout << "z_ " << endl << z << endl;

  // Sanity check if orientation offset is too big
  VectorXd y = z - z_pred;
  if(y(1)>M_PI){
    y(1) -= 2 * M_PI;
  }

  //cout << "y_ " << endl << y << endl;
  MatrixXd JHt = JH.transpose();
  MatrixXd S = JH * P_ * JHt + R_radar_;
  //cout << "S_ " << endl << S << endl;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * JHt;
  MatrixXd K = PHt * Si;
  //cout << "K_ " << endl << K << endl;

  //new estimate
  x_ = x_ + (K * y);   
  P_ = (I_ - K * JH) * P_;

  // print predicted state
  //cout << "x_updR " << endl << x_ << endl;
  //cout << "P_updR " << endl << P_ << endl;

}
