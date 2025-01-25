#ifndef MEI_VIDEOSAVER_HPP
#define MEI_VIDEOSAVER_HPP

#include "Managers/CameraManager.hpp"


class VideoSaver
{
private:
	std::vector<cv::VideoWriter> videoWriters_;

public:
	void start(std::vector<std::shared_ptr<ICameraLoader>> &cameras);

	void show(std::vector<CameraImage> &cameraImages);

	void write(std::vector<CameraImage> &cameraImages);

	void finish();
};


#endif //MEI_VIDEOSAVER_HPP