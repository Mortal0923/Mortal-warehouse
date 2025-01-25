#include "Managers/DataSender.hpp"

int DataSender::timeStamp_ = 0;

DataSender::DataSender(int devIndex)
{
#if defined(WITH_SERIAL)
	portInit(devIndex);
#endif
}

void DataSender::portInit(int devIndex)
{
	fd_ = openUartSerial(("/dev/ttyUSB" + std::to_string(devIndex)).c_str());
	if (fd_ == FAILURE)
	{
		throw std::runtime_error("Error opening serial file");
	}

	if (initUartSerial(fd_, B115200, NO_FLOW_CONTROL, 8, ONE_STOP_BIT, NO_PARITY) == FAILURE)
	{
		throw std::runtime_error("Error initialize serial port");
	}

	LOGGER(Logger::INFO, "Init serial successfully", true);
}

void DataSender::writeToBuffer(int startIndex, int dataNum, const int *inputData)
{
	for (int i = 0; i < dataNum; ++i)
	{
		dataBuffer_[i + startIndex] = inputData[i];
	}
}

void DataSender::sendData()
{
	//dataBuffer_[0] = timeStamp_++;
	unsigned char data[WORDCOUNT * 2 + 2];

	int count = 0;

	for(int i = 0;i < WORDCOUNT;i++)
	{
		if(dataBuffer_[i] == 0) count++;	
	}

	if(count == WORDCOUNT) return;
	
	data[0] = 0xaa;
	for (int i = 0; i < WORDCOUNT; ++i)
	{
		data[i * 2 + 1] = dataBuffer_[i] >> 8;
		data[i * 2 + 2] = dataBuffer_[i];
	}
	data[WORDCOUNT * 2 + 1] = 0xbb;

	if (sendUartSerial(fd_, data, WORDCOUNT * 2 + 2) == SUCCESS)
	{
		std::cout << "[Info] data:\t\t";
		for (int i: dataBuffer_)
		{
			std::cout << i << " ";
		}
		std::cout << std::endl;
	}
	else
	{
		std::cerr << "[Warning] Send data failed" << std::endl;
	}
}

DataSender::~DataSender()
{
	closeUartSerial(fd_);
}

void DataSender::get_dataBuffer()
{

	//std::cout<<"left and right"<<std::endl;
	//std::cout<<"closest_x :"<<dataBuffer_[1]<<std::endl;
	//std::cout<<"closest_y :"<<dataBuffer_[2]<<std::endl;
	// if(dataBuffer_[0] != 0)
	// {
	// 	std::cout<<"///////////////////////////"<<std::endl;
	// 	std::cout<<dataBuffer_[1]<<std::endl;
	// 	std::cout<<dataBuffer_[2]<<std::endl;
	// 	std::cout<<dataBuffer_[3]<<std::endl;
	// 	std::cout<<std::endl;
	// 	std::cout<<dataBuffer_[4]<<std::endl;
	// 	std::cout<<dataBuffer_[5]<<std::endl;
	// 	std::cout<<dataBuffer_[6]<<std::endl;
	// 	std::cout<<"///////////////////////////"<<std::endl;
	//}
	/*
	std::cout<<"farthest_x :"<<dataBuffer_[4]<<std::endl;
	std::cout<<"farthest_y :"<<dataBuffer_[5]<<std::endl;
	std::cout<<"farthest_z :"<<dataBuffer_[6]<<std::endl;
	
	std::cout<<"front and back"<<std::endl;
	std::cout<<"closest_x :"<<dataBuffer_[7]<<std::endl;
	std::cout<<"closest_y :"<<dataBuffer_[8]<<std::endl;
	std::cout<<"closest_z :"<<dataBuffer_[9]<<std::endl;
	
	std::cout<<"farthest_x :"<<dataBuffer_[10]<<std::endl;
	std::cout<<"farthest_y :"<<dataBuffer_[11]<<std::endl;
	std::cout<<"farthest_z :"<<dataBuffer_[12]<<std::endl;
	*/
}

