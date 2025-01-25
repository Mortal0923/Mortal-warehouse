#pragma once

#include "Loaders/ICameraLoader.hpp"
#include "Entity/Parameters.hpp"


class RsCameraLoader :
		public ICameraLoader
{
private:
	int getFrameFromHardware(RsFrameData &frameData);

	int reconnect();

	float pixelOffset_[17][2] = {{0,  0},
	                             {0,  3},
	                             {0,  -3},
	                             {3,  0},
	                             {-3, 0},
	                             {3,  3},
	                             {-3, -3},
	                             {3,  -3},
	                             {-3, 3},
	                             {0,  5},
	                             {0,  -5},
	                             {5,  0},
	                             {-5, 0},
	                             {5,  5},
	                             {-5, -5},
	                             {5,  -5},
	                             {-5, 5}};
	cv::Mat_<float> pitchRotateMatrix_ = cv::Mat_<float>(3, 3);
	cv::Mat_<float> yawRotateMatrix_ = cv::Mat_<float>(3, 3);

	rs2::pipeline pipe_;
	rs2::config config_;
	rs2::align alignToColor_ = rs2::align(RS2_STREAM_COLOR);
	rs2::frameset currentFrameSet_;

	std::mutex queueMutex_;
	std::queue<RsFrameData> frameQueue_;

public:
	int cameraId_;
	int cameraType_;
	int imageWidth_;
	int imageHeight_;
	int framerate_;
	bool isRunning_ = true;
	Parameters parameters_;
	std::string serialNumber_;

	RsCameraLoader(int cameraId, int cameraType, int imgWidth, int imgHeight, int framerate, Parameters parameters, std::string serialNumber);

	int cameraId() override;

	int cameraType() override;

	int imageWidth() override;

	int imageHeight() override;

	void init() override;

	int start() override;

	void updateFrame() override;

	int getCurrentFrame(long currentTimeStamp, cv::Mat &colorImage) override;

	cv::Point3f getCameraPosition(const cv::Point2f &graphCenter);

	void stop() override;
};