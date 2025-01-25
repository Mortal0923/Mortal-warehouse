#if defined(WITH_CUDA)

#include "Util/MeiLogger.hpp"

MeiLogger::MeiLogger(ILogger::Severity severity) : severity_(severity)
{}

void MeiLogger::log(ILogger::Severity severity, const char *msg) noexcept
{
	if (severity <= severity_)
	{
		LOGGER(Logger::NO_TYPE, msg, false);
	}
}

#endif