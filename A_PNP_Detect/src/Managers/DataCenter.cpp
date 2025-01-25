#include "Managers/DataCenter.hpp"
// 三维二次曲线拟合函数
void quadraticFit3D(const Vector3DList& points, Eigen::Vector3d& a, Eigen::Vector3d& b, Eigen::Vector3d& c) {
    int n = points.size();
    assert(n >= 3 && "At least 3 points are required for quadratic fitting");

    Eigen::MatrixXd A(n, 3);
    Eigen::MatrixXd B(n, 3);

    // 构造矩阵 A 和矩阵 B
    for (int i = 0; i < n; ++i) {
        double x = points[i](0);
        A(i, 0) = x * x;
        A(i, 1) = x;
        A(i, 2) = 1;
        B.row(i) = points[i];
    }

    // 使用最小二乘法求解参数矩阵 [a, b, c]
    Eigen::MatrixXd result = A.colPivHouseholderQr().solve(B);

    // 提取拟合参数
    a = result.row(0);
    b = result.row(1);
    c = result.row(2);
}

// 根据已知的z值，使用牛顿法迭代求出最佳的x和y值
// 根据已知的z值，使用牛顿法迭代求出最佳的x和y值
Eigen::Vector2d solveXYFromZ(double z, const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c, double initial_x, double initial_y, int max_iterations = 100, double tolerance = 1e-6) {
    double x_optimized = initial_x;
    double y_optimized = initial_y;
    double best_x = x_optimized;
    double best_y = y_optimized;
    double min_error = std::numeric_limits<double>::max();

    for (int iter = 0; iter < max_iterations; ++iter) {
        // 计算 t 的值
        double A = a(2);
        double B = b(2);
        double C = c(2) - z;
        double discriminant = B * B - 4 * A * C;
        if (discriminant < 0) 
			return Eigen::Vector2d(-1, -1);

        double t1 = (-B + std::sqrt(discriminant)) / (2 * A);
        double t2 = (-B - std::sqrt(discriminant)) / (2 * A);

        // 选择合理的 t 值，这里简单选择较小的正数解
        double t = min(t1, t2);

        // 使用牛顿法迭代更新 x 和 y
        double f_x = a(0) * t * t + b(0) * t + c(0) - x_optimized;
        double f_y = a(1) * t * t + b(1) * t + c(1) - y_optimized;

        // 计算雅可比矩阵 (偏导数)
        double df_dt_x = 2 * a(0) * t + b(0);
        double df_dt_y = 2 * a(1) * t + b(1);

        // 更新 t 值
        double t_update = -(f_x * df_dt_x + f_y * df_dt_y) / (df_dt_x * df_dt_x + df_dt_y * df_dt_y);
        t += t_update;

        // 更新 x 和 y 值
        double new_x = a(0) * t * t + b(0) * t + c(0);
        double new_y = a(1) * t * t + b(1) * t + c(1);

        // 计算误差
        double error = std::abs(new_x - x_optimized) + std::abs(new_y - y_optimized);

        // 检查收敛条件
        if (error < tolerance) {
            best_x = new_x;
            best_y = new_y;
            break;
        }

        // 更新最佳值（如果当前误差更小）
        if (error < min_error) {
            min_error = error;
            best_x = new_x;
            best_y = new_y;
        }

        // 更新 x 和 y 值
        x_optimized = new_x;
        y_optimized = new_y;
    }

    return Eigen::Vector2d(best_x, best_y);
}

// Perform PCA and return the projection matrix and mean
void performPCA(const std::vector<Eigen::Vector3d>& points, Eigen::Matrix3d& projectionMatrix, Eigen::Vector3d& mean) {
    size_t N = points.size();

    // Compute the mean
    mean = Eigen::Vector3d::Zero();
    for (const auto& point : points) {
        mean += point;
    }
    mean /= N;

    // Center the data
    Eigen::MatrixXd centered(N, 3);
    for (size_t i = 0; i < N; ++i) {
        centered.row(i) = points[i] - mean;
    }

    // Compute the covariance matrix
    Eigen::Matrix3d covariance = centered.transpose() * centered / (N - 1);

    // Perform Eigen decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(covariance);
    projectionMatrix = eigenSolver.eigenvectors(); // Columns are the principal directions
}

// Project points onto the first two principal components
std::vector<Eigen::Vector2d> projectTo2D(const std::vector<Eigen::Vector3d>& points, const Eigen::Matrix3d& projectionMatrix, const Eigen::Vector3d& mean) {
    std::vector<Eigen::Vector2d> projectedPoints;
    for (const auto& point : points) {
        Eigen::Vector3d centeredPoint = point - mean;
        Eigen::Vector2d reducedPoint = projectionMatrix.leftCols(2).transpose() * centeredPoint;
        projectedPoints.push_back(reducedPoint);
    }
    return projectedPoints;
}

// Fit a 2D quadratic curve: y = ax^2 + bx + c
Eigen::Vector3d fit2DQuadraticCurve(const std::vector<Eigen::Vector2d>& points) {
    size_t N = points.size();
    Eigen::MatrixXd A(N, 3);
    Eigen::VectorXd b(N);

    for (size_t i = 0; i < N; ++i) {
        double x = points[i](0);
        double y = points[i](1);
        A(i, 0) = x * x;
        A(i, 1) = x;
        A(i, 2) = 1.0;
        b(i) = y;
    }

    // Solve for coefficients
    return A.colPivHouseholderQr().solve(b);
}

// Project fitted 2D curve back to 3D
std::vector<Eigen::Vector3d> reconstruct3D(const std::vector<Eigen::Vector2d>& points2D, const Eigen::Matrix3d& projectionMatrix, const Eigen::Vector3d& mean) {
    std::vector<Eigen::Vector3d> reconstructedPoints;
    for (const auto& point2D : points2D) {
        Eigen::Vector3d projectedPoint = projectionMatrix.leftCols(2) * point2D + mean;
        reconstructedPoints.push_back(projectedPoint);
    }
    return reconstructedPoints;
}

void fitQuadraticCurve(const std::vector<Eigen::Vector3d>& points, Eigen::VectorXd& coefficients) {
    size_t N = points.size();
    if (N < 10) {
        throw std::runtime_error("At least 10 points are required to fit a quadratic curve.");
    }

    // Construct matrix A
    Eigen::MatrixXd A(N, 10);
    for (size_t i = 0; i < N; ++i) {
        double x = points[i](0);
        double y = points[i](1); //s y coordinate
        double z = points[i](2); // Acceoordinate
        A(i, 0) = x * x;
        A(i, 1) = y * y;
        A(i, 2) = z * z;
        A(i, 3) = x * y;
        A(i, 4) = x * z;
        A(i, 5) = y * z;
        A(i, 6) = x;
        A(i, 7) = y;
        A(i, 8) = z;
        A(i, 9) = 1.0;
    }

    // Perform SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    coefficients = svd.matrixV().col(9); // The last column of V corresponds to the smallest singular value
}
void DataCenter::kalman_init()
{
	kalman_ball.init(Z_N, X_N);
	kalman_basket.init(Z_N, X_N);
	
	Eigen::MatrixXd H(2, 4);
	H<< 1, 0, 0, 0,
    	0, 1, 0, 0;
		
	Eigen::MatrixXd P(4, 4);
 	P << 20, 0, 0, 0,
       	0, 20, 0, 0,
	   	0, 0, 100, 0,
		0, 0, 0, 100;
	
	Eigen::MatrixXd Q(4, 4);
	Q << generateGaussianInRange(0, 1, -20, 20), 0, 0, 0,
       0, generateGaussianInRange(0, 1, -20, 20), 0, 0, 
       0, 0, generateGaussianInRange(0, 1, -20, 20), 0,
       0, 0, 0, generateGaussianInRange(0, 1, -0.1, 0.1);
	
	Eigen::MatrixXd R(2, 2);
	R << 100, 0,
         0, 100;
	   
	kalman_ball.setH(H);
	kalman_ball.setP(P);
	kalman_ball.setQ(Q);
	kalman_ball.setR(R);

	kalman_basket.setH(H);
	kalman_basket.setP(P);
	kalman_basket.setQ(Q);
	kalman_basket.setR(R);
}
void getMostConfidentBall(std::vector<Ball> &balls, Ball& mostconfidentball)
{
	for(vector<Ball>::iterator it = balls.begin(); it != balls.end(); ++it)
	{
		if(mostconfidentball.confidence_ < it->confidence_)
			mostconfidentball = *it;
	}
}

void DataCenter::setInput(IEngineLoader &engineLoader)
{
	Mat frame;
	for (CameraImage &cameraImage: cameraImages_)
	{
		//remap(cameraImage.colorImage_, cameraImage.colorImage_, pnpManager_.mapX, pnpManager_.mapY,INTER_LINEAR);
		//frame = cameraImage.colorImage_.clone();
		//undistort(frame, cameraImage.colorImage_, pnpManager_.camera_matrix, pnpManager_.dist_coeffs, pnpManager_.newCameraMatrix);
		engineLoader.setInput(cameraImage.colorImage_, cameraImage.cameraId_);
	}
}

void DataCenter::getBallData(IEngineLoader &engineLoader)
{
	for (CameraImage &cameraImage: cameraImages_)
	{
		engineLoader.getBallsByCameraId(
				cameraImage.cameraId_,
				cameraImage.cameraType_ & FRONT_CAMERA ? frontBalls_ : backBalls_
		);
	}
}

void DataCenter::FindBallCenter()
{
	if(cameraImages_.empty())
	{
		detectedball_ = false;
		return;
	}
	Mat frame = cameraImages_.back().colorImage_;
	if(MostConfidentBall.isValid_)
	{
		detectedball_ = true;
		Rect ball_rec = MostConfidentBall.graphRect();
		int rows = ball_rec.height;
		int cols = ball_rec.width;
		int ball_x = ball_rec.x;
		int ball_y = ball_rec.y;
		
		// 添加边界检查
        ball_x = std::max(0, ball_x);
        ball_y = std::max(0, ball_y);
        cols = std::min(cols, frame.cols - ball_x);
        rows = std::min(rows, frame.rows - ball_y);

        // 检查矩形是否有效
        if (cols <= 0 || rows <= 0) {
            std::cout << "Invalid ROI size. Skipping processing." << std::endl;
            return;
        }	
				
		Mat src = frame(cv::Rect(ball_x, ball_y , cols, rows));
		Mat hsv = Mat::zeros(src.size(), CV_8UC3);
		Mat mask = Mat::zeros(src.size(), CV_8UC1);
		Mat mask1 = Mat::zeros(src.size(), CV_8UC1);
		Mat mask2 = Mat::zeros(src.size(), CV_8UC1);
		// 将图像转换为HSV颜色空间
		cvtColor(src, hsv, COLOR_BGR2HSV);

		//imshow("src",hsv);
		
		cv::Scalar lower_orange1(150, 100, 0);  // 较暗的橙色
        cv::Scalar upper_orange1(179, 255, 200);  // 较亮的橙色
                    
		cv::Scalar lower_orange2(0, 100, 40);  // 较暗的橙色
        cv::Scalar upper_orange2(10, 200, 200);  // 较亮的橙色
                    
        // 创建掩码以提取橙色对象
        cv::inRange(hsv, lower_orange1, upper_orange1, mask1);
    	cv::inRange(hsv, lower_orange2, upper_orange2, mask2);
		cv::bitwise_or(mask1,mask2, mask);
		
		cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    	cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
		cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    	cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

		//imshow("mask",mask);
		
        // 查找轮廓
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;

        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		
        bool targetDetected = false;
		int maxAreaIndex_ = 0;
		int min_R = std::min(cols, rows);
        double maxArea = min_R * min_R * M_PI;
        int maxAreaIndex = 0;
		double tempArea = 0;

        // 遍历所有找到的轮廓，并选取最大面积的轮廓
        for (size_t i = 0; i < contours.size(); i++)
        {
            double area = contourArea(contours[i]);
            if (area > 500 && area < maxArea)
            {
				if(area > tempArea)
                {
					maxAreaIndex = i;
                	targetDetected = true;
					tempArea = area;
                }
            }
        }

        // 如果没有检测到目标，显示消息
        if (!targetDetected)
            return;

        // 获取最大轮廓并拟合最小的圆
        vector<Point> allPoints = contours[maxAreaIndex];

        if (!allPoints.empty())
        {
            minEnclosingCircle(allPoints, BallCenter_graphic, Ballradius_graphic);
            BallCenter_graphic.x += ball_x;
            BallCenter_graphic.y += ball_y;
        }
        else
        {
            std::cout << "No Point detected!" << std::endl;
        }
    }
	
	else
		detectedball_ = false;
}

void DataCenter::processFrontData()
{/*
	if(!frontBalls_.empty())
	{
		
		Ball& ballIt = frontBalls_.front();
		const double basketball_R = 110.01;
		vector<Point2d> imagePosition;
		Point2d ball_LA = Point2d(ballIt.graphRect().x, ballIt.graphRect().y);
		Point2d ball_RA = Point2d(ballIt.graphRect().x + ballIt.graphRect().width, ballIt.graphRect().y);
		Point2d ball_LB = Point2d(ballIt.graphRect().x, ballIt.graphRect().y + ballIt.graphRect().height);
		Point2d ball_RB = Point2d(ballIt.graphRect().x + ballIt.graphRect().width, ballIt.graphRect().y + ballIt.graphRect().height);
		//Point2d ball_Center = ballIt.graphCenter();
		imagePosition.push_back(ball_LA);
		imagePosition.push_back(ball_RA);
		imagePosition.push_back(ball_LB);
		imagePosition.push_back(ball_RB);
		//imagePosition.push_back(ball_Center);

		vector<Point3d> objectPosition;
		Point3d ball_LA_w = Point3d(0, -basketball_R, 0);
		Point3d ball_RA_w = Point3d(0, basketball_R, 0);
		Point3d ball_LB_w = Point3d(-basketball_R, 0, 0);
		//Point3d ball_Right_w = Point3d(basketball_R, 0, 0);
		Point3d ball_RB_w = Point3d(0, 0, -basketball_R);
		objectPosition.push_back(ball_Top_w);
		objectPosition.push_back(ball_Below_w);
		objectPosition.push_back(ball_Left_w);
		//objectPosition.push_back(ball_Right_w);
		objectPosition.push_back(ball_Center_w);

		
	
		const double basketball_R = 122.5;
		vector<Point2d> imagePosition;
		Point2d ball_Top = Point2d(BallCenter_graphic.x, BallCenter_graphic.y - Ballradius_graphic);
		Point2d ball_Below = Point2d(BallCenter_graphic.x, BallCenter_graphic.y + Ballradius_graphic);
		Point2d ball_Left = Point2d(BallCenter_graphic.x - Ballradius_graphic, BallCenter_graphic.y);
		//Point2d ball_Right = Point2d(BallCenter_graphic.x + Ballradius_graphic, BallCenter_graphic.y);
		Point2d ball_Center = Point2d(BallCenter_graphic.x, BallCenter_graphic.y);
		imagePosition.push_back(ball_Top);
		imagePosition.push_back(ball_Below);
		imagePosition.push_back(ball_Left);
		//imagePosition.push_back(ball_Right);
		imagePosition.push_back(ball_Center);

		vector<Point3d> objectPosition;
		Point3d ball_Top_w = Point3d(0, -basketball_R, 0);
		Point3d ball_Below_w = Point3d(0, basketball_R, 0);
		Point3d ball_Left_w = Point3d(-basketball_R, 0, 0);
		//Point3d ball_Right_w = Point3d(basketball_R, 0, 0);
		Point3d ball_Center_w = Point3d(0, 0, -basketball_R);
		objectPosition.push_back(ball_Top_w);
		objectPosition.push_back(ball_Below_w);
		objectPosition.push_back(ball_Left_w);
		//objectPosition.push_back(ball_Right_w);
		objectPosition.push_back(ball_Center_w);

		pnpManager_.setImagePoints(imagePosition);
		pnpManager_.setObjectPoints(objectPosition);

		pnpManager_.Solve();
		
		Mat rotation_vector, translation_vector;

		pnpManager_.getRotationAndTranslation(rotation_vector, translation_vector);
		
		Mat Rvec;
		Mat_<float> Tvec;
		rotation_vector.convertTo(Rvec, CV_32F);  // 旋转向量转换格式
		translation_vector.convertTo(Tvec, CV_32F); // 平移向量转换格式 

		Mat_<float> rotMat(3, 3);
		Rodrigues(Rvec, rotMat);
		// 旋转向量转成旋转矩阵

		Mat P_oc;
		P_oc = -rotMat.inv() * Tvec;
		// 求解相机的世界坐标，得出p_oc的第三个元素即相机到物体的距离即深度信息，单位是mm
		PNPPosition.x = (P_oc.at<float>(0 ,0));
		PNPPosition.y = (P_oc.at<float>(0, 1));
		PNPPosition.z =  P_oc.at<float>(0, 2);

		std::cout<<"PNPPosition:"<<PNPPosition.x<<" "<<PNPPosition.y<<" "<<PNPPosition.z<<std::endl;
		
	}
	*/
}

void DataCenter::processBackData(std::vector<std::shared_ptr<ICameraLoader>> &cameras)
{
	if(!backBalls_.empty())
	{
		//选出框
		for (auto ballIt = backBalls_.begin(); ballIt != backBalls_.end();)
		{
			if (ballIt->labelNum_ == BASKET)
			{
				Baskets_.push_back(*ballIt);
				ballIt = backBalls_.erase(ballIt);
			}
			else
			{
				ballIt++;
			}
		}
		//处理篮球数据
		processBallData(backBalls_, MostConfidentBall, lastMostConfidentBall, ball_graphic_v, kalman_ball, lastTimeStamp_ball, dt_ball, Basketball_R, Basketball_R, cameras, true);
		//cout<<"MostConfidentBall.cameraPosition: "<<MostConfidentBall.cameraPosition().x<<" "<<MostConfidentBall.cameraPosition().y<<" "<<MostConfidentBall.cameraPosition().z<<endl;
		//处理篮筐数据
		processBallData(Baskets_, MostConfidentBasket, lastMostConfidentBasket, basket_graphic_v, kalman_basket, lastTimeStamp_basket, dt_basket, Basket_HalfWidth, Basket_HalfHeight, cameras, false);
		MostConfidentBasket.offsetToEncodingDisk(cameras);
	}
}
void DataCenter::processBallData(vector<Ball>& backBalls_, Ball& MostConfidentBall, Ball& lastMostConfidentBall, float* ball_graphic_v, Kalmanfilter& kalman_ball, long& lastTimeStamp_ball, int& dt, const double& halfwidth, const double& halfheight, std::vector<std::shared_ptr<ICameraLoader>> &cameras, bool isBall)
{
	//选出置信度最高的球
	if(!backBalls_.empty())
	{
		MostConfidentBall = backBalls_.front();
		getMostConfidentBall(backBalls_, MostConfidentBall);
		if(MostConfidentBall.confidence_ > 0.5)
		{
			MostConfidentBall.isValid_ = true;

			if(lastMostConfidentBall.isValid_)
			{
				long currentTimeStamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				dt = currentTimeStamp - lastTimeStamp_ball;
				//std::cout<<"dt:"<<dt<<std::endl;
				if(dt < 2000)
				{
					double dx,dy,vx,vy,ax,ay;

					dx = MostConfidentBall.graphCenter().x - lastMostConfidentBall.graphCenter().x;
					dy = MostConfidentBall.graphCenter().y - lastMostConfidentBall.graphCenter().y;
					
					vx = dx / dt;
					vy = dy / dt;
					
					ax = (vx - ball_graphic_v[0]) / dt;
					ay = (vy - ball_graphic_v[1]) / dt;
					
					Eigen::MatrixXd X(X_N, 1);
					Eigen::MatrixXd F(X_N, X_N);
					Eigen::MatrixXd R(Z_N, Z_N);
					Eigen::MatrixXd H(Z_N, X_N);
					Eigen::MatrixXd Q(X_N, X_N);
					Eigen::VectorXd U(X_N, 1);

					F << 1, 0, dt, 0,
						0, 1, 0, dt,
						0, 0, 1, 0, 
						0, 0, 0, 1;
					
					U << 0.5 * pow(dt, 2) * ax,
						0.5 * pow(dt, 2) * ay,
						ax * dt,
						ax * dt;
						
					X << lastMostConfidentBall.graphCenter().x,
						lastMostConfidentBall.graphCenter().y,
						vx,
						vy;
						
					ball_graphic_v[0] = vx;
					ball_graphic_v[1] = vy;
					
					kalman_ball.setF(F);
					kalman_ball.setX(X);
					kalman_ball.setU(U);

					kalman_ball.predict();
					Eigen::MatrixXd Z(2, 1);
					Z << MostConfidentBall.graphCenter().x,
						MostConfidentBall.graphCenter().y;
					kalman_ball.Measurement_update(Z);
					Eigen::VectorXd final_X = kalman_ball.getX();
					
					MostConfidentBall.ballPositions_.front().graphCenter_.x = final_X[0];
					MostConfidentBall.ballPositions_.front().graphCenter_.y = final_X[1];
					
					PnP_Calculate(MostConfidentBall, halfwidth, halfheight);

					MostConfidentBall.offsetToEncodingDisk(cameras);

					if(isBall)
					{
						Calculate_ball_v(MostConfidentBall, lastMostConfidentBall, ball_v, dt);
					
						if(sqrt(pow(MostConfidentBall.cameraPosition().x - lastMostConfidentBall.cameraPosition().x, 2) + pow(MostConfidentBall.cameraPosition().y - lastMostConfidentBall.cameraPosition().y, 2) + pow(MostConfidentBall.cameraPosition().z - lastMostConfidentBall.cameraPosition().z, 2)) > 20)
							Basketballpoints.push_back({MostConfidentBall.cameraPosition().x, MostConfidentBall.cameraPosition().y, MostConfidentBall.cameraPosition().z});
						angle_theta_basketball = acos(MostConfidentBall.cameraPosition().x / sqrt(pow(MostConfidentBall.cameraPosition().x, 2) + pow(MostConfidentBall.cameraPosition().y, 2) + pow(MostConfidentBall.cameraPosition().z, 2))) * 180 / M_PI;

					}
					

					lastMostConfidentBall = MostConfidentBall;

					lastTimeStamp_ball = currentTimeStamp;
				}
				else
				{
					
					lastMostConfidentBall.isValid_ = false;

					for(int i = 0;i < 2;i++)
						ball_graphic_v[i] = 0;
					
					//临时使用，后期记得区分球和篮筐
					angle_theta_basketball = acos(MostConfidentBall.cameraPosition().x / sqrt(pow(MostConfidentBall.cameraPosition().x, 2) + pow(MostConfidentBall.cameraPosition().y, 2) + pow(MostConfidentBall.cameraPosition().z, 2))) * 180 / M_PI;

					lastTimeStamp_ball = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				}
			}

			else
			{
				PnP_Calculate(MostConfidentBall, halfwidth, halfheight);

				MostConfidentBall.offsetToEncodingDisk(cameras);
				
				if(isBall)
				{
					Basketballpoints.push_back({MostConfidentBall.cameraPosition().x, MostConfidentBall.cameraPosition().y, MostConfidentBall.cameraPosition().z});
					angle_theta_basketball = acos(MostConfidentBall.cameraPosition().x / sqrt(pow(MostConfidentBall.cameraPosition().x, 2) + pow(MostConfidentBall.cameraPosition().y, 2) + pow(MostConfidentBall.cameraPosition().z, 2))) * 180 / M_PI;
				}			
				lastMostConfidentBall = MostConfidentBall;

				for(int i = 0;i < 2;i++)
					ball_graphic_v[i] = 0;

				lastTimeStamp_ball = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
			}
		}

		else
		{
			MostConfidentBall.isValid_ = false;
		}
	}
}

void DataCenter::Calculate_ball_v(Ball& MostConfidentBall, Ball& lastMostConfidentBall, double* ball_v, int& dt)
{
	if(lastMostConfidentBall.isValid_)
	{
		double dx_w, dy_w, dz_w, vx_w, vy_w, vz_w, ax_w, ay_w, az_w;

		dx_w = MostConfidentBall.ballPositions_.front().cameraPosition_.x - lastMostConfidentBall.ballPositions_.front().cameraPosition_.x;
		dy_w = MostConfidentBall.ballPositions_.front().cameraPosition_.y - lastMostConfidentBall.ballPositions_.front().cameraPosition_.y;
		dz_w = MostConfidentBall.ballPositions_.front().cameraPosition_.z - lastMostConfidentBall.ballPositions_.front().cameraPosition_.z;
		
		vx_w = dx_w / dt;
		vy_w = dy_w / dt;
		vz_w = dz_w / dt;

		ax_w = (vx_w - ball_v[0]) / dt;
		ay_w = (vy_w - ball_v[1]) / dt;
		az_w = (vz_w - ball_v[2]) / dt;

		ball_v[0] = vx_w;
		ball_v[1] = vy_w;
		ball_v[2] = vz_w;
	}

	else
	{
		for(int i = 0;i < 3;i++)
			ball_v[i] = 0;
	}
}

void DataCenter::setSenderBuffer(DataSender &dataSender)
{
	int backData[4] = {0};

	if(MostConfidentBall.isValid_)
	{
		backData[0] = predicted_point2d.x;
		backData[1] = predicted_point2d.y;
		backData[2] = MostConfidentBall.cameraPosition().z;
		backData[3] = angle_theta_basketball * 10;
	}

	dataSender.writeToBuffer(0, 4, backData);
}

void DataCenter::printData(ostream& outfile_x, ostream& outfile_y, ostream& outfile_z, ostream& outfile_vx, ostream& outfile_vy, ostream& outfile_vz)
{
	if(MostConfidentBall.isValid_)
	{	
		std::cout<<"----------------------------------------------------------"<<std::endl;
		std::cout << "BallPosition: " << MostConfidentBall.cameraPosition().x << " "<< MostConfidentBall.cameraPosition().y << " "<< MostConfidentBall.cameraPosition().z << std::endl;
		std::cout<<"Ball_v: " << ball_v[0] << " " << ball_v[1] << " " << ball_v[2] << std::endl;
		std::cout<<"angle: "<< angle_theta_basketball << std::endl;
		outfile_x << "{"<<MostConfidentBall.cameraPosition().x <<","<< MostConfidentBall.cameraPosition().y <<","<< MostConfidentBall.cameraPosition().z <<"},"<<endl;
		double target_z = 122.5;

		// if(predict(target_z))
		// {
		// 	std::cout<<"predict: "<<predicted_point2d.x<<" "<<predicted_point2d.y<<std::endl;
		// 	outfile_x << "predict: "<<predicted_point2d.x <<", "<< predicted_point2d.y <<endl;
		// }
		// outfile_x << MostConfidentBall.cameraPosition().x <<endl;
		// outfile_y << MostConfidentBall.cameraPosition().y <<endl;
		// outfile_z << MostConfidentBall.cameraPosition().z <<endl;
		// outfile_vx << ball_v[0] <<endl;
		// outfile_vy << ball_v[1] <<endl;z
		// outfile_vz << ball_v[2] <<endl; 

		// double actual_y = (MostConfidentBall.cameraPosition().y  + 1000) / 1000.0;
		// double actual_z = (MostConfidentBall.cameraPosition().z) / 1000.0;
		// double vy = ball_v[1];
		// double vz = ball_v[2];



		// // std::cout<<"actual_y: "<<actual_y<<std::endl;
		// // std::cout<<"actual_z: "<<actual_z<<std::endl;
		// //  std::cout<<"vy: "<<vy<<std::endl;
		// //  std::cout<<"vz: "<<vz<<std::endl;

		// predicted_z = actual_z + (pow(abs(pow(vy, 2) - 2 * 9.8 * actual_y), 0.5) - vy) * vz / 9.8;

		// predicted_z *= 1000;
		// std::cout<<"predict_z: "<<predicted_z<<std::endl;

	}
	if(MostConfidentBasket.isValid_)
		std::cout << "BasketPosition: " << MostConfidentBasket.cameraPosition().x << " "<< MostConfidentBasket.cameraPosition().y << " "<< MostConfidentBasket.cameraPosition().z << std::endl;

	
}

void DataCenter::drawFrontImage()
{
cv::Mat *images[3] = {nullptr};
	for (CameraImage &cameraImage: cameraImages_)
	{
		images[cameraImage.cameraId_] = &cameraImage.colorImage_;
	}

	for (int i = 0; i < frontBalls_.size(); ++i)
	{
		Ball &tempBall = frontBalls_.at(i);
		for (const BallPosition &ballPosition: tempBall.ballPositions_)
		{
			if (images[ballPosition.cameraId_])
			{
				cv::Mat &img = *images[ballPosition.cameraId_];
				rectangle(img, MostConfidentBall.graphRect(), RED, 2);
				rectangle(img, MostConfidentBasket.graphRect(), RED, 2);
				/*
				putText(img, std::to_string(tempBall.labelNum_) + (tempBall.isInBasket_ ? " B " : " G ") + std::to_string(i),
				        cv::Point2i(ballPosition.graphRect_.x, ballPosition.graphRect_.y),
				        cv::FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2);
				putText(img, "x: " + std::to_string(ballPosition.cameraPosition_.x).substr(0, 6),
				        cv::Point2i(ballPosition.graphRect_.x, ballPosition.graphRect_.y + 12),
				        cv::FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1);
				putText(img, "y: " + std::to_string(ballPosition.cameraPosition_.y).substr(0, 6),
				        cv::Point2i(ballPosition.graphRect_.x, ballPosition.graphRect_.y + 24),
				        cv::FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1);
				putText(img, "z: " + std::to_string(ballPosition.cameraPosition_.z).substr(0, 6),
				        cv::Point2i(ballPosition.graphRect_.x, ballPosition.graphRect_.y + 36),
				        cv::FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1);
				*/
				if(detectedball_)
					circle(img, BallCenter_graphic, Ballradius_graphic, Scalar(0, 255, 0), 1);
			}
		}
	}
	if (!frontBalls_.empty())
	{
		for (const BallPosition &ballPosition: frontBalls_.front().ballPositions_)
		{
			if (images[ballPosition.cameraId_])
			{
				cv::Mat &img = *images[ballPosition.cameraId_];
				rectangle(img, ballPosition.graphRect_, GREEN, 2);
			}
		}
	}
	if (frontBalls_.size() >= 2)
	{
		for (const BallPosition &ballPosition: frontBalls_.at(1).ballPositions_)
		{
			if (images[ballPosition.cameraId_])
			{
				cv::Mat &img = *images[ballPosition.cameraId_];
				rectangle(img, ballPosition.graphRect_, WHITE, 2);
			}
		}
	}
}

void DataCenter::drawBackImage()
{
	cv::Mat *images[3] = {nullptr};
	for (CameraImage &cameraImage: cameraImages_)
	{
		images[cameraImage.cameraId_] = &cameraImage.colorImage_;
	}

	for (int i = 0; i < backBalls_.size(); ++i)
	{
		Ball &tempBall = backBalls_.at(i);
		for (const BallPosition &ballPosition: tempBall.ballPositions_)
		{
			if (images[ballPosition.cameraId_])
			{
				cv::Mat &img = *images[ballPosition.cameraId_];
				if(MostConfidentBall.isValid_)
				{
					rectangle(img, MostConfidentBall.graphRect(), GREEN, 2);
					putText(img, std::to_string(MostConfidentBall.labelNum_) + " Basketball " + std::to_string(MostConfidentBall.confidence_),
							cv::Point2i(MostConfidentBall.graphRect().x, MostConfidentBall.graphRect().y),
							cv::FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2);
				}

				if(MostConfidentBasket.isValid_)
				{
					rectangle(img, MostConfidentBasket.graphRect(), RED, 2);
					putText(img, std::to_string(MostConfidentBasket.labelNum_) + " Basket " + std::to_string(MostConfidentBasket.confidence_),
							cv::Point2i(MostConfidentBasket.graphRect().x, MostConfidentBasket.graphRect().y),
							cv::FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2);
				}
			}
		}
	}
}

void DataCenter::clearAll()
{
	cameraImages_.clear();
	
	frontBalls_.clear();
	backBalls_.clear();
	
	Baskets_.clear();

	MostConfidentBall.isValid_ = false;
	MostConfidentBasket.isValid_ = false;
}

void DataCenter::PnP_Calculate(Ball& MostConfidentBall, const double& objection_halfwidth, const double& objection_halfheight)
{
	if(MostConfidentBall.isValid_)
	{
		vector<Point2d> imagePosition;
		vector<Point3d> objectPosition;
		vector<Point2d> undisort_imagePosition;

		double width = MostConfidentBall.graphRect().width;
		double height = MostConfidentBall.graphRect().height;

		Point2d ball_TopLeft = Point2d(MostConfidentBall.graphCenter().x - width/2, MostConfidentBall.graphCenter().y - height/2);
		Point2d ball_TopRight = Point2d(MostConfidentBall.graphCenter().x + width/2, MostConfidentBall.graphCenter().y - height/2);
		Point2d ball_BelowLeft = Point2d(MostConfidentBall.graphCenter().x - width/2, MostConfidentBall.graphCenter().y + height/2);
		Point2d ball_BelowRight = Point2d(MostConfidentBall.graphCenter().x + width/2, MostConfidentBall.graphCenter().y + height/2);

		imagePosition.push_back(ball_TopLeft);
		imagePosition.push_back(ball_TopRight);
		imagePosition.push_back(ball_BelowLeft);
		imagePosition.push_back(ball_BelowRight);

		//std::cout << "imagePosition: " << imagePosition[0].x<<" "<< imagePosition[0].y<< std::endl;
		undistortImagePoints(imagePosition, undisort_imagePosition, pnpManager_.camera_matrix, pnpManager_.dist_coeffs);
		//std::cout << "undisort_imagePosition: " << undisort_imagePosition[0].x<<" "<< undisort_imagePosition[0].y<< std::endl;

		width = abs(undisort_imagePosition[0].x - undisort_imagePosition[1].x);
		height = abs(undisort_imagePosition[0].y - undisort_imagePosition[2].y);

		//std::cout<<"width: "<<width<<" height: "<<height<<std::endl;

		width = min(width, height);
		height = width;

		Point2d basketball_center = Point2d((undisort_imagePosition[0].x + undisort_imagePosition[1].x)/2.0, (undisort_imagePosition[0].y + undisort_imagePosition[2].y)/2.0);

		imagePosition.clear();
		ball_TopLeft = Point2d(basketball_center.x - width/2, basketball_center.y - height/2);
		ball_TopRight = Point2d(basketball_center.x + width/2, basketball_center.y - height/2);
		ball_BelowLeft = Point2d(basketball_center.x - width/2, basketball_center.y + height/2);
		ball_BelowRight = Point2d(basketball_center.x + width/2, basketball_center.y + height/2);
		imagePosition.push_back(ball_TopLeft);
		imagePosition.push_back(ball_TopRight);
		imagePosition.push_back(ball_BelowLeft);
		imagePosition.push_back(ball_BelowRight);
		
		Point3d ball_TopLeftw = Point3d(-objection_halfwidth, objection_halfheight, 0);
		Point3d ball_TopRightw = Point3d(objection_halfwidth, objection_halfheight, 0);
		Point3d ball_BelowLeftw = Point3d(-objection_halfwidth, -objection_halfheight, 0);
		Point3d ball_BelowRightw = Point3d(objection_halfwidth, -objection_halfheight, 0);

		objectPosition.push_back(ball_TopLeftw);
		objectPosition.push_back(ball_TopRightw);
		objectPosition.push_back(ball_BelowLeftw);
		objectPosition.push_back(ball_BelowRightw);

		pnpManager_.setImagePoints(imagePosition);
		pnpManager_.setObjectPoints(objectPosition);

		pnpManager_.Solve();

		Mat rotation_vector, translation_vector;

		pnpManager_.getRotationAndTranslation(rotation_vector, translation_vector);

		Mat Rvec;
		Mat_<float> Tvec;
		rotation_vector.convertTo(Rvec, CV_32F);  // 旋转向量转换格式
		translation_vector.convertTo(Tvec, CV_32F); // 平移向量转换格式 

		Mat_<float> rotMat(3, 3);
		Rodrigues(Rvec, rotMat);
		// 旋转向量转成旋转矩阵

		Mat P_oc;
		P_oc = -rotMat.inv() * Tvec;
		// 求解相机的世界坐标，得出p_oc的第三个元素即相机到物体的距离即深度信息，单位是mm， 并更改为世界三维坐标系
		MostConfidentBall.ballPositions_.front().cameraPosition_.x = -(P_oc.at<float>(0 ,0));
		MostConfidentBall.ballPositions_.front().cameraPosition_.y = (P_oc.at<float>(0, 2));
		MostConfidentBall.ballPositions_.front().cameraPosition_.z =  -(P_oc.at<float>(0, 1));
	}
}

bool DataCenter::predictBallPosition(Vector3DList& points, Point2d& predict_point2d, double& targetZ, double& initial_x, double& initial_y)
{
	if(points.size() < 15)
	{
		predicted_point2d.x = -1;
		predicted_point2d.y = -1;
		return false;
	}

	else
	{
    	Eigen::Matrix3d projectionMatrix;
    	Eigen::Vector3d mean;
    	performPCA(points, projectionMatrix, mean);

		// Project to 2D
		std::vector<Eigen::Vector2d> projectedPoints = projectTo2D(points, projectionMatrix, mean);

		// Fit a 2D quadratic curve
		Eigen::Vector3d quadraticCoeffs = fit2DQuadraticCurve(projectedPoints);
		//std::cout << "Fitted 2D quadratic curve coefficients: " << quadraticCoeffs.transpose() << std::endl;

		// Evaluate curve in 2D
		std::vector<Eigen::Vector2d> curve2D;
		for (double x = -1.0; x <= 6.0; x += 0.1) {
			double y = quadraticCoeffs(0) * x * x + quadraticCoeffs(1) * x + quadraticCoeffs(2);
			curve2D.emplace_back(x, y);
		}

		// Reconstruct 3D curve
		std::vector<Eigen::Vector3d> reconstructedCurve = reconstruct3D(curve2D, projectionMatrix, mean);

		// Output reconstructed 3D points
		// for (const auto& point : reconstructedCurve) {
		// 	std::cout << "Reconstructed 3D point: " << point.transpose() << std::endl;
		// }

		Eigen::Vector3d a, b, c;
		quadraticFit3D(points, a, b, c);
		
		// 根据已知的z值求出对应的x和y值（包含牛顿法迭代优化）
		Eigen::Vector2d xy = solveXYFromZ(targetZ, a, b, c, initial_x, initial_y);
		predict_point2d = {xy(0), xy(1)};

		points.clear();

		return true;
	}
}

bool DataCenter::predict(double& targetZ, double& initial_x, double& initial_y)
{
	return predictBallPosition(Basketballpoints, predicted_point2d, targetZ, initial_x, initial_y);
}