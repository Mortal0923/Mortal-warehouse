#pragma once

#include <iostream>
#include "Util/serial.hpp"

class DataSender
{
private:
	int fd_;
	static int timeStamp_;
	/// \brief number of words in A data packet, excluding start and end of packet
	static constexpr int WORDCOUNT = 4;
	/// \brief	[0]time stamp start from 0
	/// 			[1]ball flag
	/// 			[2-5]first nearest ball, including position in camera coordinate system(x, y, z) and label
	/// 			[6-9]second nearest ball
	/// 			[10-14]ball labels of row0, which counts from bottom to top, left to right
	/// 			[15-19]ball labels of row1
	/// 			[20-24]ball labels of row2
	int dataBuffer_[WORDCOUNT];

public:
	explicit DataSender(int devIndex);

	void portInit(int devIndex);

	void writeToBuffer(int startIndex, int dataNum, const int *inputData);

	void sendData();

	~DataSender();
	
	void get_dataBuffer();
};
