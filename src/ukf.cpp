#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() 
{
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    n_x_ = 5;

    // initial state vector
    x_ = VectorXd(n_x_);

    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);
    P_.setIdentity();

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 3;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 3;

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


    is_initialized_ = false;

    n_aug_ = n_x_ + 2;

    lambda_ = 3.0 - n_aug_;

    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    Xsig_pred_.fill(0.0);

    weights_ = VectorXd(2 * n_aug_ + 1);
    weights_.fill(0.5/(lambda_ + n_aug_));
    weights_(0) = lambda_ / (lambda_ + n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage & meas_package) 
{

    if (!is_initialized_)
    {
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) 
        {
            x_ = polarToCartesian(meas_package.raw_measurements_);
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) 
        {
            x_ << meas_package.raw_measurements_[0], 
                meas_package.raw_measurements_[1], 
                0,
                0, 
                0;
        }

        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }

    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0; //dt - expressed in seconds
    time_us_ = meas_package.timestamp_;

    

    if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && use_laser_)
    {
        Prediction(dt);
        UpdateLidar(meas_package);
    }
    else if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) && use_radar_)
    {
        Prediction(dt);
        UpdateRadar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) 
{

    MatrixXd sigmaPoints;
    generateSigmaPoints(&sigmaPoints);

    predictSigmaPoints(delta_t, sigmaPoints);

    predictMeanAndCov();

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const MeasurementPackage & meas_package) 
{
    VectorXd predZ;
    MatrixXd predP;
    MatrixXd sigmaPointsZ;

    predictLaserMeasurement(&predZ, &predP, &sigmaPointsZ);

    updateState(predZ, predP, sigmaPointsZ, meas_package);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage & meas_package) 
{

    VectorXd predZ;
    MatrixXd predP;
    MatrixXd sigmaPointsZ;

    predictRadarMeasurement(&predZ, &predP, &sigmaPointsZ);

    updateState(predZ, predP, sigmaPointsZ, meas_package);

}

VectorXd 
UKF::polarToCartesian(const VectorXd & inPolar)
{
    assert(inPolar.size() >= 2);
    
    VectorXd output(5);

    double rho = inPolar(0);
    double phi = inPolar(1);

    double px = rho * std::cos(phi);
    double py = -rho * std::sin(phi);

    output << px, py, 0.0, 0.0, 0.0;

    return output;

}

void 
UKF::normalizeAngle(double & inAngle)
{
    while (inAngle > M_PI)
    {
        inAngle -= 2. * M_PI;
    } 
    while (inAngle < -M_PI) 
    {
        inAngle += 2. * M_PI;
    }
}

void 
UKF::generateSigmaPoints(MatrixXd * outSigmaPoints)
{
    VectorXd x_aug = VectorXd(n_aug_);
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    //create augmented mean state
    x_aug.head(n_x_) = x_;
    x_aug(n_x_) = 0;
    x_aug(n_x_ + 1) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug(5,5) = std_a_ * std_a_;
    P_aug(6,6) = std_yawdd_ * std_yawdd_;

    //create square root matrix
    MatrixXd A = P_aug.llt().matrixL();

    MatrixXd Xsig = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    Xsig.col(0) = x_aug;
    MatrixXd B = A * sqrt(lambda_ + n_aug_);
    B.colwise() += x_aug;
    Xsig.block(0, 1, n_aug_, n_aug_) = B;

    B = A * sqrt(lambda_ + n_aug_);
    B.colwise() -= x_aug;
    Xsig.block(0, n_aug_ + 1, n_aug_, n_aug_) = -B;

    *outSigmaPoints = Xsig;
}

void 
UKF::predictSigmaPoints(double inDt, const MatrixXd & inSigmaPoints)
{
    //predict sigma points
    for (int i(0); i< inSigmaPoints.cols(); ++i)
    {
        //extract values for better readability
        double p_x  = inSigmaPoints(0,i);
        double p_y  = inSigmaPoints(1,i);
        double v    = inSigmaPoints(2,i);
        double yaw  = inSigmaPoints(3,i);
        double yawd = inSigmaPoints(4,i);
        double nu_a = inSigmaPoints(5,i);
        double nu_yawdd = inSigmaPoints(6,i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) 
        {
            px_p = p_x + v/yawd * ( sin (yaw + yawd * inDt) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw + yawd * inDt) );
        }
        else 
        {
            px_p = p_x + v * inDt * cos(yaw);
            py_p = p_y + v * inDt * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * inDt;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5 * nu_a * inDt * inDt * cos(yaw);
        py_p = py_p + 0.5 * nu_a * inDt * inDt * sin(yaw);
        v_p = v_p + nu_a * inDt;

        yaw_p = yaw_p + 0.5 * nu_yawdd * inDt * inDt;
        yawd_p = yawd_p + nu_yawdd * inDt;

        //write predicted sigma point into right column
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
    }
}

void 
UKF::predictMeanAndCov()
{
    //predict state mean
    x_ = Xsig_pred_ * weights_;

    //predict state covariance matrix
    P_.fill(0.0);
    for (int i(0); i < Xsig_pred_.cols(); ++i) 
    {  
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        
        normalizeAngle(x_diff(3));

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }
}

void 
UKF::predictRadarMeasurement(
        VectorXd * outZ,
        MatrixXd * outCov,
        MatrixXd * outSigmaPointsZ)
{
    //set measurement dimension, radar can measure r, phi, and r_dot
    int nZ = 3;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(nZ, 2 * n_aug_ + 1);

    //mean predicted measurement
    VectorXd predZ = VectorXd(nZ);

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(nZ, nZ);

    //transform sigma points into measurement space
    for (int i(0); i < Xsig_pred_.cols(); ++i)
    {
        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);
        double v  = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        Zsig(0, i) = sqrt(pow(px, 2.0) + pow(py, 2.0));

        if (Zsig(0, i) < 0.0001)
        {
          std::cout << "Unable to calculate points in measurement space." << std::endl;
          return;
        }

        Zsig(1, i) = atan2(py, px);
        Zsig(2, i) = (px * cos(yaw) * v + py * sin(yaw) * v) / Zsig(0, i);
    }
  
    //calculate mean predicted measurement
    predZ = Zsig * weights_;

    //calculate measurement covariance matrix S
    S.fill(0.0);
    for (int i(0); i < Zsig.cols(); ++i) 
    {  
        // state difference
        VectorXd z_diff = Zsig.col(i) - predZ;

        normalizeAngle(z_diff(1));

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    S(0,0) += std_radr_ * std_radr_;
    S(1,1) += std_radphi_ * std_radphi_;
    S(2,2) += std_radrd_ * std_radrd_;

    *outZ = predZ;
    *outSigmaPointsZ = Zsig;
    *outCov = S;

}

void 
UKF::predictLaserMeasurement(
        VectorXd * outZ,
        MatrixXd * outCov,
        MatrixXd * outSigmaPointsZ)
{
    //set measurement dimension, lidar can measure px and py
    int nZ = 2;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = Xsig_pred_.block(0, 0, nZ, Xsig_pred_.cols());

    //mean predicted measurement
    VectorXd predZ = VectorXd(nZ);

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(nZ, nZ);
  
    //calculate mean predicted measurement
    predZ = Zsig * weights_;

    //calculate measurement covariance matrix S
    S.fill(0.0);
    for (int i(0); i < Zsig.cols(); ++i) 
    {  
        // state difference
        VectorXd z_diff = Zsig.col(i) - predZ;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    S(0,0) += std_laspx_ * std_laspx_;
    S(1,1) += std_laspy_ * std_laspy_;
    

    *outZ = predZ;
    *outSigmaPointsZ = Zsig;
    *outCov = S;
}

void 
UKF::updateState(
        const VectorXd & inZ,
        const MatrixXd & inCov,
        const MatrixXd & inSigmaPointsZ,
        const MeasurementPackage & meas_package)
{
    VectorXd z = meas_package.raw_measurements_;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, inZ.size());

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i(0); i < inSigmaPointsZ.cols(); ++i)
    {
        Tc = Tc + weights_(i) * (Xsig_pred_.col(i) - x_) * ((inSigmaPointsZ.col(i) - inZ).transpose());
    }
    //calculate Kalman gain K;
    MatrixXd K = Tc * inCov.inverse();

    //update state mean and covariance matrix
    x_ = x_ + K * (z - inZ);
    P_ = P_ - K * inCov * K.transpose();
}
