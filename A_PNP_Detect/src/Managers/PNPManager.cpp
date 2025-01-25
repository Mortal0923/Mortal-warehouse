#include "Managers/PNPManager.hpp"

PNPManager::PNPManager()
{
	camera_matrix = (Mat_<double>(3, 3) << 644.5697079407406, 0.0, 637.7443896505747, 0.0, 644.9706288787152, 357.53424234254567, 0.0, 0.0, 1.0);
    
    dist_coeffs = (Mat_<double>(5, 1) << -0.03999215544426365, 0.028423536276917242, -0.0007354797613712582, 0.000353478207461849, 0.0);
    
    newCameraMatrix = getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, {1280, 720}, 1);

    perspectiveMatrix = (Mat_<double>(3, 4) << 642.5234403263795, 0.0, 638.5449358525206, 0.0, 0.0, 643.0257312528474, 356.69321663539944, 0.0, 0.0, 0.0, 1.0, 0.0);
    
    initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(), perspectiveMatrix, {1280, 720}, CV_16SC2, mapX, mapY);
}

PNPManager::~PNPManager()
{
    image_points.clear();
    object_points.clear();
}

void PNPManager::setObjectPoints(vector<cv::Point3d> objectPoints)
{
    this->object_points.assign(objectPoints.begin(), objectPoints.end());
    //std::cout<<"finish set"<<std::endl;
}

void PNPManager::setImagePoints(vector<cv::Point2d> imagePoints)
{
    this->image_points.assign(imagePoints.begin(), imagePoints.end());
    //std::cout<<"finish set"<<std::endl;
}

void PNPManager::Solve()
{
    solvePnP(this->object_points, this->image_points, this->camera_matrix, this->dist_coeffs, this->rotation_vector, this->translation_vector, 0, SOLVEPNP_IPPE);
    //solvePnPRansac(this->object_points, this->image_points, this->camera_matrix, this->dist_coeffs, this->rotation_vector, this->translation_vector, false, 100, 8.0, 0.99, noArray(), SOLVEPNP_ITERATIVE);
}

void PNPManager::getRotationAndTranslation(Mat &rotation_vector, Mat &translation_vector)
{
    rotation_vector = this->rotation_vector.clone();
    translation_vector = this->translation_vector.clone();
}