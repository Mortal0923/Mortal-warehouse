#pragma once

#include "opencv2/opencv.hpp"
#include "librealsense2/rs.hpp"
#include "Util/Logger.hpp"

#define LOGGER(infoType, message, io) Logger::getInstance().writeMsg(infoType, message, io)

#define GREEN                                                        cv::Scalar(0, 255, 0)
#define WHITE                                                        cv::Scalar(255, 255, 255)
#define RED                                                            cv::Scalar(0, 0, 255)

#define ROBOT_WIDTH_LIMIT                                   275.0f
#define RADIUS                                                       95.0f

#define RS_FRAME_TIME_OUT                                   5000
#define TRACK_FRAME_TIME_OUT                              1000
#define TRACK_DELAY                                               400
#define MAX_FRAME_QUEUE_SIZE                              5
#define MAX_RECONNECT_ATTEMPTS_COUNT              3
#define MAX_INTERRUPT_COUNT                                3

struct RsFrameData
{
	long timeStamp_{};
	rs2::frameset frameset_;
};
struct CvFrameData
{
	long timeStamp_{};
	cv::Mat frame_;
};

struct CameraImage
{
	int cameraId_;
	int cameraType_;
	cv::Mat colorImage_;
};

struct LetterboxParameter
{
	int imageWidth_;
	int imageHeight_;
	int resizedWidth_;
	int resizedHeight_;
	int offsetX_;
	int offsetY_;
	float imageScale_;
};

enum OriginalLabel
{
	BASKET = 0,
	BASKETBALL = 1,
};

enum FlowControlFlag
{
	NO_FLOW_CONTROL = 0,
	HARDWARE_FLOW_CONTROL = 1,
	SOFTWARE_FLOW_CONTROL = 2,
};

enum ParityBitFlag
{
	NO_PARITY = 0,
	ODD_PARITY = 1,
	EVEN_PARITY = 2,
};

enum StopBitFlag
{
	ONE_STOP_BIT = 1,
	TWO_STOP_BIT = 2,
};

enum StatusCode
{
	FAILURE = -1,
	SUCCESS = 0,
	EMPTY_FRAME = 1,
	NO_FRAME = 2,
	TIME_OUT = 3,
};

enum CameraType
{
	FRONT_RS_CAMERA = 1 << 0,
	FRONT_WF_CAMERA = 1 << 1,
	BACK_CAMERA = 1 << 2,
	FRONT_CAMERA = FRONT_RS_CAMERA | FRONT_WF_CAMERA,
	//FRONT_CAMERA = FRONT_WF_CAMERA,
};