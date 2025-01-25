#pragma once

#include "opencv2/opencv.hpp"
#include "Entity/Parameters.hpp"
#include "Util/Functions.hpp"


class BallPosition
{
public:
	int cameraId_;
	float confidence_;
	bool isValid_;
	cv::Rect2f graphRect_;
	cv::Point2f graphCenter_;
	cv::Point3f cameraPosition_;

	BallPosition(float centerX, float centerY, float width, float height, float confidence, int cameraId);

	void setCameraPosition(cv::Point3f cameraPosition);

	void toMillimeter();

	void offsetToEncodingDisk(Parameters parameters);

	float calcDistance() const;
};