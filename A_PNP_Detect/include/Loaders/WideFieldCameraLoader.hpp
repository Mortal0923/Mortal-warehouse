#pragma once

#include "Loaders/ICameraLoader.hpp"


class WideFieldCameraLoader :
		public ICameraLoader
{
private:
	int getFrameFromHardware(CvFrameData &frameData);

	int reconnect();

	cv::VideoCapture cap_;
	std::mutex queueMutex_;
	std::queue<CvFrameData> frameQueue_;

public:
	int cameraId_;
	int cameraType_;
	int imageWidth_;
	int imageHeight_;
	int framerate_;
	bool isRunning_ = true;
	int devIndex_;

	explicit WideFieldCameraLoader(int cameraId, int cameraType, int devIndex, int imageWidth, int imageHeight, int framerate);

	int cameraId() override;

	int cameraType() override;

	int imageWidth() override;

	int imageHeight() override;

	void init() override;

	int start() override;

	void updateFrame() override;

	int getCurrentFrame(long currentTimeStamp, cv::Mat &colorImage) override;

	void stop() override;
};