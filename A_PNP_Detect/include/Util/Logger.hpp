#pragma once

#include <iostream>
#include <fstream>
#include <format>
#include <chrono>


class Logger
{
private:
	std::ofstream outStream_;
	std::string infoTypeName_[4] = {"", "[Info]", "[Warning]", "[Error]"};

	Logger();

public:
	enum InfoType
	{
		NO_TYPE = 0,
		INFO = 1,
		WARNING = 2,
		ERROR = 3
	};

	Logger(const Logger &) = delete;

	Logger &operator=(const Logger &) = delete;

	static Logger &getInstance();

	void writeMsg(int infoType, const std::string &message, bool consoleIO);

	~Logger();
};