#pragma once

#include "opencv2/opencv.hpp"
#include "Entity/Ball.hpp"
#include "Util/Functions.hpp"

class IEngineLoader
{
public:
	virtual void setInput(cv::Mat &BGRImage, int imageId) = 0;

	virtual void setInput(uint8_t *rawInput, int imageId) = 0;

	virtual void preProcess() = 0;

	virtual void infer() = 0;

	virtual void postProcess() = 0;

	virtual void getBallsByCameraId(int cameraId, std::vector<Ball> &container) = 0;

	virtual ~IEngineLoader() = default;
};