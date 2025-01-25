#pragma once

#include "define.hpp"

class Parameters
{
public:
	float XOffsetToDisk_;
	float YOffsetToDisk_;
	float ZOffsetToDisk_;
	float pitchAngle_;
	float yawAngle_;
	float changeRate_;

	Parameters();

	Parameters(float XOffsetToDisk, float YOffsetToDisk, float ZOffsetToDisk, float pitchAngle, float yawAngle, float changeRate);
};