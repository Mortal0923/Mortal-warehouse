#ifndef MEI_ICAMERALOADER_HPP
#define MEI_ICAMERALOADER_HPP

#include <thread>
#include <queue>
#include <atomic>
#include <mutex>
#include "define.hpp"


class ICameraLoader
{
public:
	virtual int cameraId() = 0;

	virtual int cameraType() = 0;

	virtual int imageWidth() = 0;

	virtual int imageHeight() = 0;

	virtual void init() = 0;

	virtual int start() = 0;

	virtual void updateFrame() = 0;

	virtual int getCurrentFrame(long currentTimeStamp, cv::Mat &colorImage) = 0;

	virtual void stop() = 0;
};


#endif //MEI_ICAMERALOADER_HPP