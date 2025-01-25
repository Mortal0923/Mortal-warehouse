#include <csignal>
#include <dlfcn.h>
#include "Loaders/TrtEngineLoader.hpp"
#include "Managers/DataCenter.hpp"
#include "Managers/CameraManager.hpp" 
#include "Managers/VideoSaver.hpp"

std::atomic<int> interruptCount = 0;

void signalHandler(int signal)
{
	interruptCount++;
	LOGGER(Logger::WARNING, std::format("Received signal {}", signal), true);
	if (interruptCount >= MAX_INTERRUPT_COUNT)
	{
		exit(-1);
	}
}

int mainBody()
{
	dlopen("libtorch_cuda.so", RTLD_NOW);

	std::ios::sync_with_stdio(false);
	std::cout.tie(nullptr);

	auto dataSender = DataSender(0);

	CameraManager cameraManager;
	cameraManager.initRsCamera();
	//cameraManager.initWFCamera();

	DataCenter dataCenter;
	dataCenter.kalman_init();

	VideoSaver videoSaver;
	videoSaver.start(cameraManager.cameras_);

	TrtEngineLoader engineLoader = TrtEngineLoader("Basket_Basketball.engine", cameraManager.cameraCount_);

	cameraManager.startUpdateThread();

	std::ofstream outfile_x("basketball_x.txt"), outfile_y("basketball_y.txt"), outfile_z("basketball_z.txt");
	std::ofstream outfile_vx("basketball_vx.txt"), outfile_vy("basketball_vy.txt"), outfile_vz("basketball_vz.txt"); 
	if(!outfile_x.is_open() || !outfile_y.is_open() || !outfile_z.is_open() || !outfile_vx.is_open() || !outfile_vy.is_open() || !outfile_vz.is_open())
	{
		std::cout<<"fail to open"<<std::endl;
	}

	double targetZ = 122.5;	
	double initial_x = 0;
	double initial_y = 0;

	while (!interruptCount)
	{
		//清除数据
		dataCenter.clearAll();

		//获取图像
		cameraManager.getCameraImage(dataCenter.cameraImages_);

		//向模型输入图片
		dataCenter.setInput(engineLoader);

		//模型预处理
		engineLoader.preProcess();

		//模型推理
		engineLoader.infer();

		//模型后处理（得到数据）
		engineLoader.postProcess();

		//获取原始数据：篮球、篮筐
		dataCenter.getBallData(engineLoader);

		//处理USB摄像头数据		
		//dataCenter.processFrontData();

		//处理realsense摄像头数据
		dataCenter.processBackData(cameraManager.cameras_);

		dataCenter.predict(targetZ,initial_x,initial_y);
		//在ROI区域找寻篮球中心
		//dataCenter.FindBallCenter();

		//查看篮球、篮筐数据
		//dataCenter.printData(outfile_x, outfile_vy, outfile_z, outfile_vx, outfile_vy, outfile_vz);
		

		//设置传输数据
		dataCenter.setSenderBuffer(dataSender);


		
#if defined(WITH_SERIAL)
		dataSender.sendData();
#endif
		//dataCenter.drawFrontImage();
		dataCenter.drawBackImage();

#if defined(GRAPHIC_DEBUG)
		videoSaver.show(dataCenter.cameraImages_);
#endif
#if defined(SAVE_VIDEO)
		videoSaver.write(dataCenter.cameraImages_);
#endif

#if defined(GRAPHIC_DEBUG)
		if (cv::waitKey(1) == 27)
		{
			break;
		}
#endif
	}
	std::cout << "Exiting. Please wait a minute..." << std::endl;

#if defined(GRAPHIC_DEBUG)
	cv::destroyAllWindows();
#endif
	cameraManager.stopUpdateThread();
	videoSaver.finish();
	
	return 0;
}

int main()
{
	signal(SIGHUP, signalHandler);//1
	signal(SIGINT, signalHandler);//2
	signal(SIGQUIT, signalHandler);//3
	signal(SIGILL, signalHandler);
	signal(SIGTRAP, signalHandler);
	signal(SIGABRT, signalHandler);//6
	signal(SIGFPE, signalHandler);//8
	signal(SIGKILL, signalHandler);//9
	signal(SIGBUS, signalHandler);//10
	signal(SIGSEGV, signalHandler);//11
	signal(SIGSYS, signalHandler);
	signal(SIGPIPE, signalHandler);
	signal(SIGALRM, signalHandler);
	signal(SIGTERM, signalHandler);//15
	signal(SIGURG, signalHandler);
	signal(SIGSTOP, signalHandler);//17
	signal(SIGTSTP, signalHandler);
	signal(SIGCONT, signalHandler);
	signal(SIGCHLD, signalHandler);
	signal(SIGTTIN, signalHandler);
	signal(SIGTTOU, signalHandler);
	signal(SIGPOLL, signalHandler);
	signal(SIGXCPU, signalHandler);
	signal(SIGXFSZ, signalHandler);
	signal(SIGVTALRM, signalHandler);
	signal(SIGPROF, signalHandler);//27

	int ret;
	try
	{
		ret = mainBody();
	}
	catch (std::exception &e)
	{
		LOGGER(Logger::ERROR, e.what(), true);
	}
	LOGGER(Logger::INFO, "Program exiting", false);
	return ret;
}
