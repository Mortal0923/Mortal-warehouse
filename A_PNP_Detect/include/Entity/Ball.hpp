#pragma once

#include "Loaders/RsCameraLoader.hpp"
#include "Entity/BallPosition.hpp"


class Ball
{
public:
	int labelNum_;
	bool isInBasket_;
	bool isValid_ = false;
	float distance_;
	float confidence_ = 0;
	std::vector<BallPosition> ballPositions_;

	int cameraId();

	cv::Point3f cameraPosition();

	cv::Rect2f graphRect();

	cv::Point2f graphCenter();

	void merge(Ball &ball);

	void addGraphPosition(float centerX, float centerY, float width, float height, float confidence, int labelNum, int cameraId, bool isInBasket);

	void setCameraPosition(std::vector<std::shared_ptr<ICameraLoader>> &cameras);

	void toMillimeter();

	void offsetToEncodingDisk(std::vector<std::shared_ptr<ICameraLoader>> &cameras);

	void calcDistance();

	bool operator<(const Ball& other) const;
};