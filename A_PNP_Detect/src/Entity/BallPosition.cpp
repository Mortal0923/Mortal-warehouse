#include "Entity/BallPosition.hpp"

BallPosition::BallPosition(float centerX, float centerY, float width, float height, float confidence, int cameraId) :
		confidence_(confidence), cameraId_(cameraId)
{
	graphCenter_ = cv::Point2f(centerX, centerY);
	graphRect_ = cv::Rect2f(centerX - width / 2, centerY - height / 2, width, height);
}

void BallPosition::setCameraPosition(cv::Point3f cameraPosition)
{
	cameraPosition_ = cameraPosition;
	isValid_ = !(cameraPosition_ == cv::Point3f(0, 0, 0));
}

void BallPosition::toMillimeter()
{
	cameraPosition_ *= 1000;
}

void BallPosition::offsetToEncodingDisk(Parameters parameters)
{
	cameraPosition_.x = parameters.changeRate_ * (cameraPosition_.x + parameters.XOffsetToDisk_);
	cameraPosition_.y = cameraPosition_.y + parameters.YOffsetToDisk_;
	cameraPosition_.z = cameraPosition_.z + parameters.ZOffsetToDisk_;
}

float BallPosition::calcDistance() const
{
	return Functions::calcDistance3f(cameraPosition_);
}
