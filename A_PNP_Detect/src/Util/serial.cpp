#include "Util/serial.hpp"

int openUartSerial(const char *port)
{
	int fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
	if (fd == -1)
	{
		LOGGER(Logger::ERROR, "Error opening serial file", false);
		return FAILURE;
	}

	//无数据则阻塞，处于等待状态
	if (fcntl(fd, F_SETFL, 0) < 0)
	{
		LOGGER(Logger::ERROR, "Error setting file status flag", false);
		return FAILURE;
	}

	//测试是否为终端设备
	if (!isatty(STDIN_FILENO))
	{
		LOGGER(Logger::ERROR, "Not a terminal device", false);
		return FAILURE;
	}
	return fd;
}

int initUartSerial(int fd, int speedCode, int flowControlFlag, int dataBits, int stopBit, int parityBit)
{
	struct termios options{};

	//得到与fd指向对象的相关参数，并将它们保存于options，该函数还可以测试配置是否正确，该串口是否可用等
	//若调用成功，函数返回值为0，若调用失败，函数返回值为1
	if (tcgetattr(fd, &options) != 0)
	{
		LOGGER(Logger::ERROR, "Error getting terminal attribute", false);
		return FAILURE;
	}

	//设置串口输入波特率和输出波特率
	cfsetispeed(&options, speedCode);
	cfsetospeed(&options, speedCode);

	//修改控制模式，保证程序不会占用串口
	options.c_cflag |= CLOCAL;
	//修改控制模式，使得能够从串口中读取输入数据
	options.c_cflag |= CREAD;

	//设置数据流控制
	switch (flowControlFlag)
	{
		case NO_FLOW_CONTROL:
			options.c_cflag &= ~CRTSCTS;
			break;
		case HARDWARE_FLOW_CONTROL:
			options.c_cflag |= CRTSCTS;
			break;
		case SOFTWARE_FLOW_CONTROL:
			options.c_cflag |= IXON | IXOFF | IXANY;
			break;
		default:    //NO_FLOW_CONTROL
			options.c_cflag &= ~CRTSCTS;
	}

	//设置数据位
	options.c_cflag &= ~CSIZE;
	switch (dataBits)
	{
		case 5:
			options.c_cflag |= CS5;
			break;
		case 6:
			options.c_cflag |= CS6;
			break;
		case 7:
			options.c_cflag |= CS7;
			break;
		case 8:
			options.c_cflag |= CS8;
			break;
		default:    //8 data bits
			options.c_cflag |= CS8;
	}

	//设置校验位
	switch (parityBit)
	{
		case NO_PARITY:
			options.c_cflag &= ~PARENB;
			options.c_iflag &= ~INPCK;
			break;
		case ODD_PARITY:
			options.c_cflag |= (PARODD | PARENB);
			options.c_iflag |= INPCK;
			break;
		case EVEN_PARITY:
			options.c_cflag |= PARENB;
			options.c_cflag &= ~PARODD;
			options.c_iflag |= INPCK;
			break;
		default:    //NO_PARITY
			options.c_cflag &= ~PARENB;
			options.c_iflag &= ~INPCK;
	}

	// 设置停止位
	switch (stopBit)
	{
		case ONE_STOP_BIT:
			options.c_cflag &= ~CSTOPB;
			break;
		case TWO_STOP_BIT:
			options.c_cflag |= CSTOPB;
			break;
		default:    //ONE_STOP_BIT
			options.c_cflag &= ~CSTOPB;
	}

	//修改输出模式，原始数据输出
	options.c_oflag &= ~OPOST;

	options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
	//options.c_lflag &= ~(ISIG | ICANON);

	//设置等待时间和最小接收字符
	options.c_cc[VTIME] = 1; /* 读取一个字符等待1*(1/10)s */
	options.c_cc[VMIN] = 1; /* 读取字符的最少个数为1 */

	//如果发生数据溢出，接收数据，但是不再读取 刷新收到的数据但是不读
	tcflush(fd, TCIFLUSH);

	//激活配置 (将修改后的termios数据设置到串口中）
	if (tcsetattr(fd, TCSANOW, &options) != 0)
	{
		LOGGER(Logger::ERROR, "Error setting terminal attribute", false);
		return FAILURE;
	}
	return SUCCESS;
}

int sendUartSerial(int fd, unsigned char *send_buf, int data_len)
{
	int len = write(fd, send_buf, data_len);
	if (len == data_len)
	{
		return SUCCESS;
	}
	else
	{
		tcflush(fd, TCOFLUSH);
		return FAILURE;
	}
}

int receiveUartSerial(int fd, char *rcv_buf, int data_len)
{
	fd_set fs_read;

	struct timeval time{};

	FD_ZERO(&fs_read);
	FD_SET(fd, &fs_read);

	time.tv_sec = 10;
	time.tv_usec = 0;

	//使用select实现串口的多路通信
	if (select(fd + 1, &fs_read, nullptr, nullptr, &time))
	{
		if (read(fd, rcv_buf, data_len) == data_len)
		{
			return SUCCESS;
		}
	}
	return FAILURE;
}

void closeUartSerial(int fd)
{
	close(fd);
}
