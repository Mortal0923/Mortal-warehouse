#ifndef MEI_DATACENTER_HPP
#define MEI_DATACENTER_HPP

#include "Loaders/IEngineLoader.hpp"
#include "Entity/Basket.hpp"
#include "Managers/DataSender.hpp"
#include "define.hpp"
#include "Managers/PNPManager.hpp"
#include "Managers/KalmanFilter.hpp"
#include <chrono>
#include <fstream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace cv;

typedef std::vector<Eigen::Vector3d> Vector3DList;

class DataCenter
{
private:

	enum OriginalLabel
	{
		BASKET = 0,
		BASKETBALL = 1,
	};

	bool haveBallInFront_ = false;
	bool detectedball_ = false;

	//滤波器参数
	const int Z_N = 2, X_N = 4;

	//识别物体的相关参数
	const double Basketball_R = 122.5;
	const double Basket_HalfWidth = 250;
	const double Basket_HalfHeight = 250;

public:
	std::vector<CameraImage> cameraImages_;
	std::vector<Ball> frontBalls_;

	PNPManager pnpManager_;			//pnp计算器

	//篮球数据
	
	std::vector<Ball> backBalls_;	//摄像头识别到的所有球
	
	Ball MostConfidentBall;			//摄像头识别到的置信度最高的球
	
	Ball lastMostConfidentBall;		//上一帧置信度最高的球
	
	float ball_graphic_v[2];		//像素坐标系下，球的速度
	
	double ball_v[3];				//相机坐标系下，球的速度
	
	Kalmanfilter kalman_ball;		//卡尔曼滤波器
	
	long lastTimeStamp_ball;		//上一帧球的时间戳

	//篮筐数据	

	std::vector<Ball> Baskets_;		//所有识别到的篮筐
	
	Ball MostConfidentBasket;		//置信度最高的篮筐
	
	Ball lastMostConfidentBasket;	//上一帧置信度最高的篮筐

	float basket_graphic_v[2];		//像素坐标系下，篮筐的速度

	double basket_v[3];				//相机坐标系下，篮筐的速度

	Kalmanfilter kalman_basket;		//卡尔曼滤波器

	long lastTimeStamp_basket;		//上一帧篮筐的时间戳
	

	Vector3DList Basketballpoints;
 	
	VectorXd coeffs; // 拟合的系数
    string modelType;


	Point2f BallCenter_graphic;	
	float Ballradius_graphic;

	int dt_ball;

	int dt_basket;

	double predicted_z;

	Point2d predicted_point2d;

	double angle_theta_basketball = 0;

	void kalman_init();
	
	void setInput(IEngineLoader &engineLoader);

	void getBallData(IEngineLoader &engineLoader);

	void processFrontData();

	void processBackData(std::vector<std::shared_ptr<ICameraLoader>> &cameras);
	
	void processBallData(vector<Ball>& backBalls_, Ball& MostConfidentBall, Ball& lastMostConfidentBall, float* ball_graphic_v, Kalmanfilter& kalman_ball, long& lastTimeStamp_ball, int& dt, const double& halfwidth, const double& halfheight, std::vector<std::shared_ptr<ICameraLoader>> &cameras, bool isBall);

	void Calculate_ball_v(Ball& MostConfidentBall, Ball& lastMostConfidentBall, double* ball_v, int& dt);

	void PnP_Calculate(Ball& MostConfidentBall, const double& objection_halfwidth, const double& objection_halfheight);
	
	void FindBallCenter();

	void setSenderBuffer(DataSender &dataSender);

	void drawFrontImage();

	void drawBackImage();

	void clearAll();

	void printData(ostream& outfile_x, ostream& outfile_y, ostream& outfile_z, ostream& outfile_vx, ostream& outfile_vy, ostream& outfile_vz);

	bool predictBallPosition(Vector3DList& points, Point2d& predict_point2d, double& targetZ,double& initial_x, double& initial_y);

	bool predict(double& targetZ, double& initial_x, double& initial_y);
};



#endif //MEI_DATACENTER_HPP