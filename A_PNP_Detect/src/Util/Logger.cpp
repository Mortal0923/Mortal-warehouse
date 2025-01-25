#include "Util/Logger.hpp"

Logger::Logger()
{
	outStream_.open("log.txt");
	if (!outStream_.is_open())
	{
		std::cerr << "[Error] Open log file failed" << std::endl;
	}
}

Logger &Logger::getInstance()
{
	static Logger instance;
	return instance;
}

void Logger::writeMsg(int infoType, const std::string &message, bool consoleIO)
{
	time_t t = time(nullptr);
	tm *tm_ = localtime(&t);
	auto now = std::chrono::system_clock::now();
	auto ms = (std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000).count();
	std::string info = std::format("[{}-{}-{} {}:{}:{}:{}]{} {}",
	                               tm_->tm_year + 1900,
	                               tm_->tm_mon + 1,
	                               tm_->tm_mday,
	                               tm_->tm_hour,
	                               tm_->tm_min,
	                               tm_->tm_sec,
	                               ms,
	                               infoTypeName_[infoType],
	                               message
	);

	if (outStream_.is_open())
	{
		outStream_ << info << "\n";
	}
	if (consoleIO)
	{
		(infoType <= INFO ? std::cout : std::cerr) << info << std::endl;
	}
}

Logger::~Logger()
{
	if (outStream_.is_open())
	{
		outStream_.close();
	}
}
