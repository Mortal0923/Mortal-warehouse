#include "Entity/Ball.hpp"
int Ball::cameraId()
{
	if (ballPositions_.empty())
	{
		throw std::runtime_error("Call cameraId() of a Ball with empty ballPositions");
	}
	return ballPositions_.front().cameraId_;
}

cv::Point3f Ball::cameraPosition()
{
	if (ballPositions_.empty())
	{
		throw std::runtime_error("Call cameraPosition() of a Ball with empty ballPositions");
	}
	return ballPositions_.front().cameraPosition_;
}

cv::Rect2f Ball::graphRect()
{
	if (ballPositions_.empty())
	{
		throw std::runtime_error("Call graphRect() of a Ball with empty ballPositions");
	}
	return ballPositions_.front().graphRect_;
}

cv::Point2f Ball::graphCenter()
{
	if (ballPositions_.empty())
	{
		throw std::runtime_error("Call graphCenter() of a Ball with empty ballPositions");
	}
	return ballPositions_.front().graphCenter_;
}

void Ball::merge(Ball &ball)
{
	for (const BallPosition &ballPosition: ball.ballPositions_)
	{
		ballPositions_.emplace_back(ballPosition);
	}
	if (ball.confidence_ > confidence_)
	{
		confidence_ = ball.confidence_;
		labelNum_ = ball.labelNum_;
		isInBasket_ = ball.isInBasket_;
	}
}

void Ball::addGraphPosition(float centerX, float centerY, float width, float height, float confidence, int labelNum, int cameraId, bool isInBasket)
{
	ballPositions_.emplace_back(centerX, centerY, width, height, confidence, cameraId);
	if (confidence > confidence_)
	{
		confidence_ = confidence;
		labelNum_ = labelNum;
		isInBasket_ = isInBasket;
	}
}

void Ball::setCameraPosition(std::vector<std::shared_ptr<ICameraLoader>> &cameras)
{
	isValid_ = false;

	for (BallPosition &ballPosition: ballPositions_)
	{
		std::shared_ptr<RsCameraLoader> RsCamera = std::static_pointer_cast<RsCameraLoader>(cameras.at(ballPosition.cameraId_));
		ballPosition.setCameraPosition(RsCamera->getCameraPosition(ballPosition.graphCenter_));
		isValid_ |= ballPosition.isValid_;
	}
}

void Ball::toMillimeter()
{
	for (BallPosition &ballPosition: ballPositions_)
	{
		ballPosition.toMillimeter();
	}
}

void Ball::offsetToEncodingDisk(std::vector<std::shared_ptr<ICameraLoader>> &cameras)
{
	for (BallPosition &ballPosition: ballPositions_)
	{
		std::shared_ptr<RsCameraLoader> RsCamera = std::static_pointer_cast<RsCameraLoader>(cameras.at(ballPosition.cameraId_));
		ballPosition.offsetToEncodingDisk(RsCamera->parameters_);
	}
}

void Ball::calcDistance()
{
	float summary = 0;
	for (BallPosition &ballPosition: ballPositions_)
	{
		summary += ballPosition.calcDistance();
	}
	distance_ = summary / ballPositions_.size();
}

bool Ball::operator<(const Ball& other) const 
{
	return this->confidence_ > other.confidence_;
}