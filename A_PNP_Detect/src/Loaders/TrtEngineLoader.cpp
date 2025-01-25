#if defined(WITH_CUDA)

#include "Loaders/TrtEngineLoader.hpp"

void TrtEngineLoader::loadEngine(std::string &enginePath)
{
	std::vector<unsigned char> modelData;

	std::ifstream inputFileStream(enginePath.c_str(), std::ios::binary);
	std::streamsize engineSize;
	if (inputFileStream.good())
	{
		inputFileStream.seekg(0, std::ifstream::end);
		engineSize = inputFileStream.tellg();
		modelData.resize(engineSize);
		inputFileStream.seekg(0, std::ifstream::beg);
		inputFileStream.read(reinterpret_cast<char *>(modelData.data()), engineSize);
		inputFileStream.close();
	}
	runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trtLogger));
	cudaEngine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(modelData.data(), engineSize));
	executionContext_ = std::unique_ptr<nvinfer1::IExecutionContext>(cudaEngine_->createExecutionContext());

	LOGGER(Logger::INFO, std::format("Load engine {} successfully", enginePath), true);
}

void TrtEngineLoader::setInOutputSize()
{
	///以本项目所用模型为例，输入batchSize*3*640*640（NCHW）
	///其中3为通道数，两个640依次为矩阵的高和宽
	auto inputDims = cudaEngine_->getTensorShape("images");
	if (inputDims.d[0] != -1)
	{
		batchSize_ = inputDims.d[0];
	}
	inputLayerHeight_ = inputDims.d[2];
	inputLayerWidth_ = inputDims.d[3];
	inputSize_ = inputDims.d[1] * inputLayerHeight_ * inputLayerWidth_;
	inputTensorSize_ = batchSize_ * inputSize_;
	
	///以本项目所用模型（YOLO8-p2）为例，输出batchSize*11*34000（NHW）
	///其中11为：centerX, centerY, width, height, clsConf0, clsConf1, ...，25200为先验框数量
	///加上TensorRT EfficientNMS Plugin后，输出分为四个部分


	auto detectedBoxesDims = cudaEngine_->getTensorShape("det_boxes");
	maxOutputNumber_ = 300;detectedBoxesDims.d[1];

	//std::cout<<"maxOutputNumber_:"<<maxOutputNumber_<<std::endl;
}

void TrtEngineLoader::initBuffers()
{
	setInOutputSize();
	//std::cout<<"finish setInOutputSize"<<std::endl;

	//letterbox
	imageScale_ = std::min((inputLayerWidth_ * 1.) / inputImageWidth_, (inputLayerHeight_ * 1.) / inputImageHeight_);
	int borderWidth = inputImageWidth_ * imageScale_;
	int borderHeight = inputImageHeight_ * imageScale_;
	offsetX_ = (inputLayerWidth_ - borderWidth) / 2;
	offsetY_ = (inputLayerHeight_ - borderHeight) / 2;

	cudaStreamCreate(&meiCudaStream_);
	cudaMalloc(&numDetBuffer_, batchSize_ * 1 * sizeof(int));
	cudaMalloc(&detBoxesBuffer_, batchSize_ * maxOutputNumber_ * 4 * sizeof(float));
	cudaMalloc(&detScoresBuffer_, batchSize_ * maxOutputNumber_ * sizeof(float));
	cudaMalloc(&detClassesBuffer_, batchSize_ * maxOutputNumber_ * sizeof(int));

	//std::cout<<"finish letterbox"<<std::endl;
	//imageBatch
	imageBatch_ = nvcv::TensorBatch(batchSize_);
	for (int i = 0; i < batchSize_; ++i)
	{
		imageBatch_.pushBack(nvcv::Tensor(1, {inputImageWidth_, inputImageHeight_}, nvcv::FMT_BGR8));
	}
	//std::cout<<"finish imageBatch"<<std::endl;
	
	//imageTensor
	imageTensor_ = nvcv::Tensor(batchSize_, {inputImageWidth_, inputImageHeight_}, nvcv::FMT_BGR8);
	//inputTensor
	inputTensor_ = nvcv::Tensor(batchSize_, {inputLayerWidth_, inputLayerHeight_}, nvcv::FMT_RGBf32p);

	//tensors in preprocess
	resizedImageTensor_ = nvcv::Tensor(batchSize_, {borderWidth, borderHeight}, nvcv::FMT_BGR8);
	rgbImageTensor_ = nvcv::Tensor(batchSize_, {borderWidth, borderHeight}, nvcv::FMT_RGB8);
	borderImageTensor_ = nvcv::Tensor(batchSize_, {inputLayerWidth_, inputLayerHeight_}, nvcv::FMT_RGB8);
	normalizedImageTensor_ = nvcv::Tensor(batchSize_, {inputLayerWidth_, inputLayerHeight_}, nvcv::FMT_RGBf32);
	
	
	
	executionContext_->setInputShape("images", nvinfer1::Dims4{batchSize_, 3, inputLayerWidth_, inputLayerHeight_});
	executionContext_->setTensorAddress("images", inputTensor_.exportData<nvcv::TensorDataStridedCuda>()->basePtr());
	
	
	
	executionContext_->setTensorAddress("num_dets", numDetBuffer_);
	executionContext_->setTensorAddress("det_boxes", detBoxesBuffer_);
	
	//std::cout<<"finish set det_boxes"<<std::endl;
	
	executionContext_->setTensorAddress("det_scores", detScoresBuffer_);
	executionContext_->setTensorAddress("det_classes", detClassesBuffer_);

	detectedBalls_.insert(detectedBalls_.begin(), batchSize_, {});
	
	//std::cout<<"finish initBuffers"<<std::endl;
}

TrtEngineLoader::TrtEngineLoader(std::string enginePath, int batchSize) : batchSize_(batchSize)
{
	initLibNvInferPlugins(&trtLogger, "");
	loadEngine(enginePath);
	initBuffers();
}

void TrtEngineLoader::setInput(cv::Mat &BGRImage, int imageId)
{
	setInput(BGRImage.data, imageId);
}

void TrtEngineLoader::setInput(uint8_t *rawInput, int imageId)
{
	if (imageId >= batchSize_)
	{
		throw std::runtime_error(std::format("ImageId {} exceeded batch size limit {}", imageId, batchSize_));
	}

	auto it = imageBatch_.begin();
	while (--imageId >= 0)
	{
		it++;
	}

	auto singleImageBuffer = it->exportData<nvcv::TensorDataStridedCuda>();
	cudaMemcpyAsync(singleImageBuffer->basePtr(), rawInput, singleImageBuffer->stride(0), cudaMemcpyHostToDevice, meiCudaStream_);
}

void TrtEngineLoader::preProcess()
{
	stack_(meiCudaStream_, imageBatch_, imageTensor_);
	//resize
	resize_(meiCudaStream_, imageTensor_, resizedImageTensor_, NVCV_INTERP_LINEAR);
	//cvtColor(BGR -> RGB)
	cvtColor_(meiCudaStream_, resizedImageTensor_, rgbImageTensor_, NVCV_COLOR_BGR2RGB);
	//copyMakeBorder
	copyMakeBorder_(meiCudaStream_, rgbImageTensor_, borderImageTensor_,
	                offsetY_, offsetX_, NVCV_BORDER_CONSTANT, {114, 114, 114, 0});
	//normalize
	convertTo_(meiCudaStream_, borderImageTensor_, normalizedImageTensor_, 1.0 / 255.0, 0);
	//reformat(NHWC -> NCHW)
	reformat_(meiCudaStream_, normalizedImageTensor_, inputTensor_);
	cudaStreamSynchronize(meiCudaStream_);
	//std::cout<<"preprocess finished"<<std::endl;
}

void TrtEngineLoader::infer()
{
	executionContext_->enqueueV3(meiCudaStream_);
	cudaStreamSynchronize(meiCudaStream_);
	//std::cout<<"infer finished"<<std::endl;
}

void TrtEngineLoader::postProcess()
{
	for (int i = 0; i < batchSize_; ++i)
	{
		detectedBalls_.at(i).clear();
	}

	torch::Tensor detectNumbers;
	torch::Tensor detectBoxes;
	torch::Tensor detectScores;
	torch::Tensor detectClasses;
	torch::Tensor basketTag;

	detectNumbers = torch::from_blob(numDetBuffer_, {batchSize_, 1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
	detectBoxes = torch::from_blob(detBoxesBuffer_, {batchSize_ * maxOutputNumber_, 4}, torch::dtype(torch::kFloat).device(torch::kCUDA));
	detectScores = torch::from_blob(detScoresBuffer_, {batchSize_, maxOutputNumber_}, torch::dtype(torch::kFloat).device(torch::kCUDA));
	detectClasses = torch::from_blob(detClassesBuffer_, {batchSize_, maxOutputNumber_}, torch::dtype(torch::kInt32).device(torch::kCUDA));

	detectBoxes.slice(1, 2, 3) = (detectBoxes.slice(1, 2, 3) - detectBoxes.slice(1, 0, 1)) / imageScale_;
	detectBoxes.slice(1, 3, 4) = (detectBoxes.slice(1, 3, 4) - detectBoxes.slice(1, 1, 2)) / imageScale_;
	detectBoxes.slice(1, 0, 1) = (detectBoxes.slice(1, 0, 1) - offsetX_) / imageScale_ + detectBoxes.slice(1, 2, 3) / 2;
	detectBoxes.slice(1, 1, 2) = (detectBoxes.slice(1, 1, 2) - offsetY_) / imageScale_ + detectBoxes.slice(1, 3, 4) / 2;

	basketTag = detectClasses % 2 == 1;
	detectClasses = (detectClasses).to(torch::kInt32);

	detectNumbers = detectNumbers.to(torch::kCPU);
	detectBoxes = detectBoxes.to(torch::kCPU);
	detectScores = detectScores.to(torch::kCPU);
	detectClasses = detectClasses.to(torch::kCPU);
	basketTag = basketTag.to(torch::kCPU);

	auto detectNumbersAccess = detectNumbers.accessor<int, 2>();
	auto detectBoxesAccess = detectBoxes.accessor<float, 2>();
	auto detectScoresAccess = detectScores.accessor<float, 2>();
	auto detectClassesAccess = detectClasses.accessor<int, 2>();
	auto basketTagAccess = basketTag.accessor<bool, 2>();

	for (int batchSize = 0; batchSize < batchSize_; ++batchSize)
	{
		for (int index = 0; index < detectNumbersAccess[batchSize][0]; ++index)
		{
			Ball ball;
			ball.addGraphPosition(
					detectBoxesAccess[batchSize * maxOutputNumber_ + index][0],
					detectBoxesAccess[batchSize * maxOutputNumber_ + index][1],
					detectBoxesAccess[batchSize * maxOutputNumber_ + index][2],
					detectBoxesAccess[batchSize * maxOutputNumber_ + index][3],
					detectScoresAccess[batchSize][index],
					detectClassesAccess[batchSize][index],
					batchSize,
					basketTagAccess[batchSize][index]
			);
			detectedBalls_.at(batchSize).push_back(ball);
		}
	}
	//std::cout<<"postprocess finished"<<std::endl;
}

void TrtEngineLoader::getBallsByCameraId(int cameraId, std::vector<Ball> &container)
{
	if (cameraId >= batchSize_)
	{
		throw std::runtime_error(std::format("cameraId {} exceeded batch size limit {}", cameraId, batchSize_));
	}

	for (const Ball &tempBall: detectedBalls_.at(cameraId))
	{
		container.push_back(tempBall);
	}
	//std::cout<<"getBallsByCameraId finished"<<std::endl;
}

TrtEngineLoader::~TrtEngineLoader()
{
	cudaStreamDestroy(meiCudaStream_);
	cudaFree(numDetBuffer_);
	cudaFree(detBoxesBuffer_);
	cudaFree(detScoresBuffer_);
	cudaFree(detClassesBuffer_);
}

#endif