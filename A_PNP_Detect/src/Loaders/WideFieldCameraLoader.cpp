#include "Loaders/WideFieldCameraLoader.hpp"

int WideFieldCameraLoader::getFrameFromHardware(CvFrameData &frameData)
{
	if (cap_.read(frameData.frame_))
	{
		if (frameData.frame_.empty())
		{
			return EMPTY_FRAME;
		}
		frameData.timeStamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		return SUCCESS;
	}
	else
	{
		return FAILURE;
	}
}

int WideFieldCameraLoader::reconnect()
{

}

WideFieldCameraLoader::WideFieldCameraLoader(int cameraId, int cameraType, int imageWidth, int imageHeight, int framerate, int devIndex) :
		cameraId_(cameraId), cameraType_(cameraType), imageWidth_(imageWidth), imageHeight_(imageHeight), framerate_(framerate), devIndex_(devIndex)
{}

void WideFieldCameraLoader::init()
{
	cap_ = cv::VideoCapture(200 + devIndex_);
	cap_.set(cv::CAP_PROP_FRAME_WIDTH, imageWidth_);
	cap_.set(cv::CAP_PROP_FRAME_HEIGHT, imageHeight_);
	cap_.set(cv::CAP_PROP_FPS, framerate_);
}

int WideFieldCameraLoader::start()
{
	return cap_.isOpened() ? SUCCESS : FAILURE;
}

void WideFieldCameraLoader::updateFrame()
{
	CvFrameData frameData;
	while (isRunning_)
	{
		int status = getFrameFromHardware(frameData);
		if (status == SUCCESS)
		{
			std::lock_guard<std::mutex> lock(queueMutex_);
			frameQueue_.push(frameData);
			while (frameQueue_.size() > MAX_FRAME_QUEUE_SIZE)
			{
				frameQueue_.pop();
			}
		}
		else if (status == FAILURE)
		{
			LOGGER(Logger::WARNING, std::format("Camera /dev/video{} disconnected", devIndex_), true);
			break;
		}
		else
		{
			LOGGER(Logger::WARNING, std::format("Empty frame in camera /dev/video{}, passed", devIndex_), true);
		}
	}
	LOGGER(Logger::INFO, std::format("Sub-thread of Camera /dev/video{} quited", devIndex_), true);
}

int WideFieldCameraLoader::getCurrentFrame(long currentTimeStamp, cv::Mat &colorImage)
{
	std::lock_guard<std::mutex> lock(queueMutex_);
	if (frameQueue_.empty())
	{
		return NO_FRAME;
	}
	else
	{
		while (true)
		{
			CvFrameData frameData = frameQueue_.front();
			frameQueue_.pop();
			if (frameData.timeStamp_ >= currentTimeStamp || frameQueue_.empty())
			{
				colorImage = frameData.frame_;
				break;
			}
			else if (frameQueue_.front().timeStamp_ >= currentTimeStamp)
			{
				if (currentTimeStamp - frameData.timeStamp_ >= frameQueue_.front().timeStamp_ - currentTimeStamp)
				{
					colorImage = frameQueue_.front().frame_;
				}
				else
				{
					colorImage = frameData.frame_;
				}
				break;
			}
		}
		return SUCCESS;
	}
}

void WideFieldCameraLoader::stop()
{
	cap_.release();
}

int WideFieldCameraLoader::cameraId()
{
	return cameraId_;
}

int WideFieldCameraLoader::cameraType()
{
	return cameraType_;
}

int WideFieldCameraLoader::imageWidth()
{
	return imageWidth_;
}

int WideFieldCameraLoader::imageHeight()
{
	return imageHeight_;
}