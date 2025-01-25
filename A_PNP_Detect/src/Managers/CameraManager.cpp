#include "Managers/CameraManager.hpp"

void CameraManager::initRsCamera()
{
	rs2::context context;
	rs2::device_list deviceList = context.query_devices();

	cameras_.reserve(3);

	if (!deviceList.size())
	{
		throw std::runtime_error("No Realsense camera detected");
	}
	else if (deviceList.size() < 3)
	{
		std::cerr << "[Warning] Detected " << deviceList.size() << " Realsense camera successfully. Please connect more cameras" << std::endl;
	}
	else
	{
		std::cout << "[Info] Detected 3 Realsense cameras successfully" << std::endl;
	}

	for (auto &&rsCamera: deviceList)
	{
		std::string serialNumber = rsCamera.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);

		auto it = paramsMap_.find(serialNumber);
		if (serialNumber == frontCameraSerialNumber_)
		{
			cameras_.push_back(
					std::make_shared<RsCameraLoader>(
							cameraCount_, FRONT_RS_CAMERA, 1280, 720, 30, Parameters(), serialNumber
					)
			);
			cameras_.back()->init();
			cameras_.back()->start();
		}
		else if (it != paramsMap_.end())
		{
			cameras_.push_back(
					std::make_shared<RsCameraLoader>(
							cameraCount_, BACK_CAMERA, 1280, 720, 30, it->second, serialNumber
					)
			);
			cameras_.back()->init();
			cameras_.back()->start();
		}
		else
		{
			LOGGER(Logger::WARNING, std::format("Detected unregistered realsense camera {}", serialNumber), true);
			continue;
		}

//		std::vector<rs2::sensor> sensors = rsCamera.query_sensors();
//		rs2::color_sensor colorSensor = rs2::color_sensor(sensors[1]);
//		colorSensor.set_option(RS2_OPTION_HUE, 10);

		LOGGER(Logger::INFO, std::format("Realsense camera {}({}) connected", serialNumber, cameraCount_), true);
		cameraCount_++;
	}
}

void CameraManager::initWFCamera()
{
	int index = 2;
	std::string info;
	v4l2_capability cap{};
	struct stat statInfo{};
	struct group *group_;

	while (index <= 20)
	{
		std::string cameraFilePath = "/dev/video" + std::to_string(index);

		if (stat(cameraFilePath.c_str(), &statInfo) == -1)
		{
			index += 2;
			continue;
		}
		group_ = getgrgid(statInfo.st_gid);
		if (std::string(group_->gr_name) != "video")
		{
			index += 2;
			continue;
		}

		int fd = open(cameraFilePath.c_str(), O_RDONLY);
		ioctl(fd, VIDIOC_QUERYCAP, &cap);
		close(fd);

		info = std::string(reinterpret_cast<char *>(cap.card));
		break;
	}

	if (info.empty())
	{
		throw std::runtime_error("No wide field camera detected");
	}
	else
	{
		cameras_.push_back(
				std::make_shared<WideFieldCameraLoader>(
						cameraCount_, FRONT_WF_CAMERA, 1920, 1080, 60, index
				)
		);
		cameras_.back()->init();
		cameras_.back()->start();

		LOGGER(Logger::INFO, std::format("Wide field camera {}({}) connected", info, cameraCount_), true);
		cameraCount_++;
	}
}

void CameraManager::startUpdateThread()
{
	for (auto &camera: cameras_)
	{
		std::thread thread(&ICameraLoader::updateFrame, camera);
		thread.detach();
	}
}

void CameraManager::getCameraImage(std::vector<CameraImage> &cameraImages)
{
	int status;
	std::string info;
	cv::Mat colorImage;

	long currentTimeStamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	for (auto &camera: cameras_)
	{
		if (camera->getCurrentFrame(currentTimeStamp, colorImage) == SUCCESS)
		{
			cameraImages.emplace_back(camera->cameraId(), camera->cameraType(), colorImage);
		}
	}
}

void CameraManager::stopUpdateThread()
{
	for (auto &camera: cameras_)
	{
		camera->stop();
	}
	std::this_thread::sleep_for(std::chrono::seconds(1));
}