#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <chrono>
#include "define.hpp"

class Kalmanfilter
{
private:

    Eigen::MatrixXd F, //状态转移矩阵  
                    Q, //过程噪声
                    H, //测量矩阵 
                    P, //状态协方差矩阵
                    R, //测量噪声矩阵
                    U; //过程影响向量
                    
    Eigen::VectorXd x; //状态向量
    
    static const int Z_N;
    static const int X_N;


public:
    Kalmanfilter();
    Kalmanfilter(const Eigen::MatrixXd& F_,
                 const Eigen::MatrixXd& Q_,
                 const Eigen::MatrixXd& H_,
                 const Eigen::MatrixXd& P_,
                 const Eigen::MatrixXd& R_);
    void setF(Eigen::MatrixXd& F_);
    void setP(Eigen::MatrixXd& P_);
    void setQ(Eigen::MatrixXd& Q_);
    void setH(Eigen::MatrixXd& H_);
    void setR(Eigen::MatrixXd& R_);
    void setU(Eigen::VectorXd& U_);
    void setX(const Eigen::VectorXd& x_);
    Eigen::VectorXd getX();
    
    void init(int Z_N_, int X_N_);
    void predict();
    void Measurement_update(const Eigen::VectorXd&z);
    
        // 平滑函数，输入为 cv::Rect2f
    cv::Rect2f smooth(const cv::Rect2f& rect);
};

double generateGaussianInRange(double mean, double stddev, double minVal, double maxVal);
