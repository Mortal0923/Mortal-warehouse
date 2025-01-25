#include "Managers/KalmanFilter.hpp"

Kalmanfilter::Kalmanfilter()
{}

double generateGaussianInRange(double mean, double stddev, double minVal, double maxVal) 
{
    // 使用默认随机引擎，可以根据需要替换为其他类型的引擎
    std::default_random_engine generator;
    // 使用当前时间作为种子以获得不同的随机数序列
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    
    // 创建高斯分布对象
    std::normal_distribution<double> distribution(mean, stddev);
    
    // 生成一个高斯分布的随机数
    double num = distribution(generator);
    
    // 确保结果在[minVal, maxVal]之间
    if (num < minVal) num = minVal;
    if (num > maxVal) num = maxVal;
    
    return num;
}
Kalmanfilter::Kalmanfilter(const Eigen::MatrixXd& F_,
               const Eigen::MatrixXd& Q_,
               const Eigen::MatrixXd& H_,
               const Eigen::MatrixXd& P_,
               const Eigen::MatrixXd& R_) :
    F(F_), Q(Q_), H(H_), P(P_), R(R_)
    {}
void Kalmanfilter::setF(Eigen::MatrixXd& F_)
{
  F = F_;
}

void Kalmanfilter::setP(Eigen::MatrixXd& P_)
{
  P = P_;
}

void Kalmanfilter::setQ(Eigen::MatrixXd& Q_)
{
  Q = Q_;
}

void Kalmanfilter::setH(Eigen::MatrixXd& H_)
{
  H = H_;
}

void Kalmanfilter::setR(Eigen::MatrixXd& R_)
{
  R = R_;
}

void Kalmanfilter::setU(Eigen::VectorXd& U_)
{
  U = U_;
}

void Kalmanfilter::setX(const Eigen::VectorXd& x_)
{
  x = x_;
}
Eigen::VectorXd Kalmanfilter::getX()
{
  return x;
}
void Kalmanfilter::init(int Z_N_, int X_N_)
{
  x = Eigen::VectorXd::Zero(X_N_);
  U = Eigen::VectorXd::Zero(X_N_);
  F = Eigen::MatrixXd::Identity(X_N_,X_N_);
  H = Eigen::MatrixXd::Zero(Z_N_,X_N_);
  
  P = Eigen::MatrixXd::Identity(X_N_,X_N_);
  /*
  P << 20, 0, 0, 0, 0, 0,
       0, 20, 0, 0, 0, 0,
       0, 0, 20, 0, 0, 0,
       0, 0, 0, 100, 0, 0,
       0, 0, 0, 0, 100, 0,
       0, 0, 0, 0, 0, 100;
  
  Q = Eigen::MatrixXd::Zero(6,6);
  R = Eigen::MatrixXd::Zero(3,3);
  
  Q << generateGaussianInRange(0, 1, -20, 20), 0, 0, 0, 0, 0,
       0, generateGaussianInRange(0, 1, -20, 20), 0, 0, 0, 0,
       0, 0, generateGaussianInRange(0, 1, -20, 20), 0, 0, 0,
       0, 0, 0, generateGaussianInRange(0, 1, -0.1, 0.1), 0, 0,
       0, 0, 0, 0,generateGaussianInRange(0, 1, -0.1, 0.1), 0,
       0, 0, 0, 0, 0, generateGaussianInRange(0, 1, -0.1, 0.1);
       
  R << 100, 0, 0,
       0, 100, 0,
       0, 0, 100;
  
  H << 1, 0, 0, 0, 0, 0,
				0, 1, 0, 0, 0, 0,
				0, 0, 1, 0, 0, 0;
  */
}
void Kalmanfilter::predict()
{
  x = F*x + U;
  //std::cout<<"物理模型预测 x:{ "<<x[0]<<" "<<x[1]<<" "<<x[2]<<" }"<<std::endl;
  P = F*P*F.transpose() + Q;
}

void Kalmanfilter::Measurement_update(const Eigen::VectorXd&z)
{
  Eigen::VectorXd y = z - H*x;  //观测值与预测值之差
  
  //std::cout<<"观测与预测值之差:"<<std::endl;
  //std::cout<<"delta x: "<<y[0]<<std::endl;
  //std::cout<<"delta y: "<<y[1]<<std::endl;
  //std::cout<<"delta z: "<<y[2]<<std::endl;
  
  Eigen::MatrixXd S = H*P*H.transpose() + R;  
  Eigen::MatrixXd K = P*H.transpose()*S.inverse();  //卡尔曼增益
  
  //std::cout<<"K:"<< K <<std::endl;
  
  x += K*y;  //更新x状态

  int size = x.size();
  P = (Eigen::MatrixXd::Identity(size,size) - K*H)*P; //更新状态协方差矩阵
}

cv::Rect2f Kalmanfilter::smooth(const cv::Rect2f& rect) {
    // 将 cv::Rect2f 转换为观测向量 z
    Eigen::VectorXd z(4);
    z << rect.x, rect.y, rect.width, rect.height;

    // 预测步骤
    predict();

    // 更新测量值
    Measurement_update(z);

    // 返回平滑后的检测框
    return cv::Rect2f(x(0), x(1), x(2), x(3));
}
