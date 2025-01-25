#pragma once

#if defined(WITH_CUDA)

#include <iostream>
#include <NvInfer.h>
#include "define.hpp"

class MeiLogger :
		public nvinfer1::ILogger
{
public:
	explicit MeiLogger(ILogger::Severity severity = ILogger::Severity::kVERBOSE);

	ILogger::Severity severity_;

	void log(ILogger::Severity severity, const char *msg) noexcept override;
};

static MeiLogger trtLogger;

#endif