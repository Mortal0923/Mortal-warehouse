#include "Util/Functions.hpp"

float Functions::calcIou(cv::Rect2f rect1, cv::Rect2f rect2)
{
	float internArea = (rect1 & rect2).area();
	float unionArea = rect1.area() + rect2.area() - internArea;
	return internArea / unionArea;
}

float Functions::calcDistance3f(cv::Point3f cameraPosition1, cv::Point3f cameraPosition2)
{
	return std::sqrt(pow(cameraPosition1.x - cameraPosition2.x, 2)
	                 + pow(cameraPosition1.y - cameraPosition2.y, 2)
	                 + pow(cameraPosition1.z - cameraPosition2.z, 2));
}

float Functions::calcDistance2f(cv::Point2f pixelPosition1, cv::Point2f pixelPosition2)
{
	return std::sqrt(pow(pixelPosition1.x - pixelPosition2.x, 2) + pow(pixelPosition1.y - pixelPosition2.y, 2));
}