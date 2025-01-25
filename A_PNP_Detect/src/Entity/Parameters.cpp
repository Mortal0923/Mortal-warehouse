#include "Entity/Parameters.hpp"

Parameters::Parameters()
{
	XOffsetToDisk_ = 0;
	YOffsetToDisk_ = 0;
	ZOffsetToDisk_ = 0;
	pitchAngle_ = 0;
	yawAngle_ = 0;
	changeRate_ = 1;
}

Parameters::Parameters(float XOffsetToDisk, float YOffsetToDisk, float ZOffsetToDisk, float pitchAngle, float yawAngle, float changeRate) :
		XOffsetToDisk_(XOffsetToDisk), YOffsetToDisk_(YOffsetToDisk), ZOffsetToDisk_(ZOffsetToDisk),
		pitchAngle_(pitchAngle), yawAngle_(yawAngle), changeRate_(changeRate)
{}
