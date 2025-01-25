#pragma once

#include <iostream>
#include <vector>
#include "define.hpp"

using namespace std;
using namespace cv;

class PNPManager
{
private:
    vector<Point2d> image_points;  // 2D点坐标
    vector<Point3d> object_points;  // 3D点坐标

public:
	Mat camera_matrix; // 相机内参矩阵
    Mat dist_coeffs;   // 相机畸变系数
    Mat rotation_vector;  // 旋转向量
    Mat translation_vector;  // 平移向量
    Mat newCameraMatrix;
    Mat mapX;
    Mat mapY;
    Mat perspectiveMatrix;
    Mat D;
    Mat K;

    PNPManager();
    ~PNPManager();
    
    void setObjectPoints(vector<cv::Point3d> objectPoints);
    void setImagePoints(vector<cv::Point2d> imagePoints);
    void Solve();
    void getRotationAndTranslation(Mat &rotation_vector, Mat &translation_vector);
};