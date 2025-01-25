#pragma once

#include "opencv2/opencv.hpp"
#include "define.hpp"

class Functions
{
public:
	static float calcIou(cv::Rect2f rect1, cv::Rect2f rect2);

	static float calcDistance3f(cv::Point3f cameraPosition1, cv::Point3f cameraPosition2 = cv::Point3f(0, 0, 0));

	static float calcDistance2f(cv::Point2f pixelPosition1, cv::Point2f pixelPosition2 = cv::Point2f(0, 0));
};