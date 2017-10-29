#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

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

double Tools::Normalize(double angle) {
  double result = angle;
  while (result > M_PI) result -= 2.*M_PI;
  while (result < -M_PI) result += 2.*M_PI;
  return result;
}
