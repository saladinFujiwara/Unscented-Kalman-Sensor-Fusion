#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) 
{
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    if (estimations.size() == 0 ||
    estimations.size() != ground_truth.size())
    {
        cout << "Invalid input vectors" << endl;
        return rmse;
    }

    //accumulate squared residuals
    VectorXd squaredSum(4);
    squaredSum << 0,0,0,0;
    
    for(int i = 0; i < estimations.size(); ++i)
    {
        
        VectorXd res = estimations[i] - ground_truth[i];
        VectorXd resSq = res.array() * res.array();
        squaredSum += resSq;
        
    }

    //calculate the mean
    squaredSum = squaredSum / estimations.size();

    //calculate the squared root
    rmse = squaredSum.array().sqrt();

    //return the result
    return rmse;  
}