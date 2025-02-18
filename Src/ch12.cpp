#include <iostream>
#include <opencv2/opencv.hpp>

int main(){
	/* OpenCV与传统机器学习 */
	/*
	OpenCV中的传统机器学习算法
	1. K-近邻算法
	2. 支持向量机
	3. 决策树
	4. 随机森林
	5. Adaboost
	6. 贝叶斯分类器
	7. EM算法
	8. K均值聚类
	9. 层次聚类
	10. 主成分分析
	11. 独立成分分析
	12. 马尔科夫随机场
	13. 条件随机场
	14. 概率图模型
	15. 深度信念网络
	16. 限制玻尔兹曼机
	17. 自组织映射
	18. 线性判别分析
	19. 非线性判别分析
	20. 朴素贝叶斯
	21. 高斯混合模型
	22. 隐马尔科夫模型
	23. 最大熵模型
	24. 逻辑回归
	25. 神经网络
	26. 深度学习
	27. 卷积神经网络
	28. 循环神经网络
	29. 生成对抗网络
	30. 强化学习
	*/
	/*
	K均值聚类算法
	cv::kmeans(	InputArray data, 				// 输入数据
				int K, 							// 聚类种类数目
				InputOutputArray bestLabels, 	// 存储每个数据聚类结果索引的矩阵或者向量
				TermCriteria criteria, 			// 终止条件
				int attempts, 					// 采样不同初始化标签尝试次数
				int flags, 						// 每类中心初始化方法标志
				OutputArray centers 			// 最终聚类后的每个类的中心位置坐标
				);								// 返回值doubel数据类型,
	
	模型训练函数
	cv::ml::StatModel::train( 	const Ptr<TrainData>& trainData, 	// 训练样本数据
								int flags=0 						// 构建模型方法标志
								);
	
	模型训练函数
	cv::ml::StatModel::train(	InputArray samples, 				// 训练样本数据
								int layout, 						// 样本数据排列方式的标志
								InputArray responses 				// 对应的响应值,对于分类是类别标签;回归是连续值
								);
	
	模型保存函数
	cv::ml::TrainData::create(	InputArray samples, 					// 训练样本数据
								int layout, 							// 样本数据排列方式的标志
								InputArray responses, 					// 对应的响应值,对于分类是类别标签;回归是连续值
								InputArray varIdx = noArray(), 			// 用于指定哪些变量用于训练的向量
								InputArray sampleIdx = noArray(), 		// 用于指定哪些样本用于训练的向量
								InputArray sampleWeights = noArray(), 	// 每个样本数据权重向量
								InputArray varType = noArray() 			// 声明变量类型的标志
								);

	模型预测函数(利用模型对新数据进行预测)
	cv::ml::StatModel::predict(	InputArray samples, 		// 训练样本数据
								OutputArray results, 		// 预测结果的输出矩阵
								int flags = 0 				// 构建模型方法标志
								);							// 返回值bool数据类型，表示是否预测的结果，true表示有，false表示无

	模型加载函数
	cv::Algorithm::load(const String& filename, 			// 文件名
						const String& objname = String() 	// 可选择的要读取节点名称
						);
		
	K近邻模型对象创建函数
	cv::KNearst::create(int defaultK = 32, 			// 默认K值
						int isRegression = false, 	// 是否回归
						int maxK = INT_MAX 			// 最大K值
						);
	
	K近邻模型训练函数
	cv::ml::KNearest::findNearest(	InputArray samples, 			// 训练样本数据
									int k, 							// K近邻的样本数目
									OutputArray results, 			// 预测结果的输出矩阵
									OutputArray neighborResponses, 	// 可以选择输出的每个数据最近邻的k个样本
									OutputArray dists 				// 可以选择输出的与k个最近邻样本的距离
									);								// 返回值float数据类型，表示距离的结果
	
	cv::ml::DTrees::create();  	// 创建决策树对象
	cv::ml::RTrees::create();  	// 创建随机森林对象
	cv::ml::SVM::create();  	// 创建SVM对象
	*/
	
	// 生成一个500×500的白色图像用于显示特征点和分类结果
	cv::Mat kmeanpoints(500, 500, CV_8UC3, cv::Scalar(255,255,255));
	cv::RNG rng(10000);

	// 设置三种颜色
	cv::Scalar colorLut1[3] = 
	{
		cv::Scalar(0, 0, 255),	// 红
		cv::Scalar(0, 255, 0),	// 绿
		cv::Scalar(255, 0, 0),	// 蓝
	};

	// 设置三个点集，并且每个点集中点的数目随机
	int number1 = 3;
	int Points1 = rng.uniform(20, 200);	// 第一类点集数量
	int Points2 = rng.uniform(20, 200);	// 第二类点集数量
	int Points3 = rng.uniform(20, 200);	// 第三类点集数量
	int Points_num = Points1 + Points2 + Points3;
	cv::Mat Points(Points_num, 1, CV_32FC2);
	
	int i = 0;
	for (; i < Points1; i++)						// 点集在(150, 150)附近
	{
		cv::Point2f pts;
		pts.x = rng.uniform(100, 200);
		pts.y = rng.uniform(100, 200);
		Points.at<cv::Point2f>(i, 0) = pts;
	}

	for (; i < Points1+ Points2; i++)				// 点集在(350, 200)附近
	{
		cv::Point2f pts;
		pts.x = rng.uniform(300, 400);
		pts.y = rng.uniform(100, 300);
		Points.at<cv::Point2f>(i, 0) = pts;
	}

	for (; i < Points1+ Points2+ Points3; i++)		// 点集在(150, 440)附近
	{
		cv::Point2f pts;
		pts.x = rng.uniform(100, 200);
		pts.y = rng.uniform(390, 490);
		Points.at<cv::Point2f>(i, 0) = pts;
	}

	// 使用KMeans
	cv::Mat labels1;  //每个点所属的种类
	cv::Mat centers1;  //每类点的中心位置坐标
	cv::kmeans(Points, number1, labels1, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1), 3, cv::KMEANS_PP_CENTERS, centers1);

	// 根据分类为每个点设置不同的颜色
	kmeanpoints = cv::Scalar::all(255);
	for (int i = 0; i < Points_num; i++)
	{
		int index = labels1.at<int>(i);
		cv::Point point = Points.at<cv::Point2f>(i);
		cv::circle(kmeanpoints, point, 2, colorLut1[index], -1, 4);
	}

	// 绘制每个聚类的中心来绘制圆
	for (int i = 0; i < centers1.rows; i++) 
	{
		int x = centers1.at<float>(i, 0);
		int y = centers1.at<float>(i, 1);
		std::cout << "第" << i + 1 << "类的中心坐标：x=" << x << "  y=" << y << std::endl;
		cv::circle(kmeanpoints, cv::Point(x, y), 50, colorLut1[i], 1, cv::LINE_AA);
	}

	cv::imshow("K近邻点集分类结果", kmeanpoints);
	cv::waitKey(0);
	cv::destroyAllWindows();


	cv::Mat people = cv::imread("../Img/people.jpg");
	if (!people.data)
	{
		printf("请确认图像文件是否输入正确");
		return -1;
	}
	
	cv::Vec3b colorLut2[5] = {
		cv::Vec3b(0, 0, 255),
		cv::Vec3b(0, 255, 0),
		cv::Vec3b(255, 0, 0),
		cv::Vec3b(0, 255, 255),
		cv::Vec3b(255, 0, 255)
	};

	// 图像的尺寸，用于计算图像中像素点的数目
	int width = people.cols;
	int height = people.rows;
	
	// 初始化定义
	int sampleCount = width * height;
	
	// 将图像矩阵数据转换成每行一个数据的形式
	cv::Mat sample_data = people.reshape(3, sampleCount);
	cv::Mat data1;
	sample_data.convertTo(data1, CV_32F);

	// KMean函数将像素值进行分类
	int number2 = 3;  // 分割后的颜色种类
	cv::Mat labels2;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1);
	cv::kmeans(data1, number2, labels2, criteria, number2, cv::KMEANS_PP_CENTERS);

	// 显示图像分割结果
	cv::Mat result2 = cv::Mat::zeros(people.size(), people.type());
	for (int row = 0; row < height; row++) 
	{
		for (int col = 0; col < width; col++) 
		{
			int index = row*width + col;
			int label = labels2.at<int>(index, 0);
			result2.at<cv::Vec3b>(row, col) = colorLut2[label];
		}
	}

	cv::imshow("原图", people);
	cv::imshow("分割后图像", result2);
	cv::waitKey(0);
	cv::destroyAllWindows();


	cv::Mat text = cv::imread("../Img/digits.png");
	cv::Mat text_gray;
	cv::cvtColor(text, text_gray, cv::COLOR_BGR2GRAY);

	// 分割为5000个cells
	cv::Mat images3 = cv::Mat::zeros(5000, 400, CV_8UC1);
	cv::Mat labels3 = cv::Mat::zeros(5000, 1, CV_8UC1);

	int index = 0;
	cv::Rect numbertext;
	numbertext.x = 0;
	numbertext.height = 1;
	numbertext.width = 400;
	for (int row = 0; row < 50; row++) 
	{
		// 从图像中分割出20×20的图像作为独立数字图像
		int label = row / 5;
		int datay = row * 20;
		for (int col = 0; col < 100; col++) 
		{
			int datax = col * 20;
			cv::Mat number = cv::Mat::zeros(cv::Size(20, 20), CV_8UC1);
			for (int x = 0; x < 20; x++) 
			{
				for (int y = 0; y < 20; y++)
				{
					number.at<uchar>(x, y) = text_gray.at<uchar>(x + datay, y + datax);
				}
			}
			// 将二维图像数据转成行数据
			cv::Mat row = number.reshape(1, 1);
			std::cout << "提取第" << index + 1 << "个数据" << std::endl;
			numbertext.y = index;
			// 添加到总数据中
			row.copyTo(images3(numbertext));
			// 记录每个图像对应的数字标签
			labels3.at<uchar>(index, 0) = label;
			index++;
		}
	}
	cv::imwrite("../Img/textimages.png", images3);
    cv::imwrite("../Img/textlabels.png", labels3);

	// 加载训练数据集
	images3.convertTo(images3, CV_32FC1);
	labels3.convertTo(labels3, CV_32SC1);
	cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(images3, cv::ml::ROW_SAMPLE, labels3);

	// 创建K近邻类
	cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();  
	knn->setDefaultK(5);  // 每个类别拿出5个数据
	knn->setIsClassifier(true);  // 进行分类
	
	// 训练数据
	knn->train(tdata);
	// 保存训练结果
	knn->save("../Data/knn_model.yml");

	// 输出运行结果提示
	std::cout << "已使用K近邻完成数据训练和保存" << std::endl;
	std::cout << std::endl;
	cv::waitKey(0);
	
	// 加载KNN分类器
	cv::Ptr<cv::ml::KNearest> knntest = cv::Algorithm::load<cv::ml::KNearest>("../Data/knn_model.yml");

	// 测试新图像是否能够识别数字
	cv::Mat handWrite01 = cv::imread("../Img/handWrite01.png", cv::IMREAD_GRAYSCALE);
	cv::Mat handWrite02 = cv::imread("../Img/handWrite02.png", cv::IMREAD_GRAYSCALE);
	cv::imshow("testImg1", handWrite01);
	cv::imshow("testImg2", handWrite02);

	// 缩放到20×20的尺寸
	cv::resize(handWrite01, handWrite01, cv::Size(20, 20));
	cv::resize(handWrite02, handWrite02, cv::Size(20, 20));
	cv::Mat testdata = cv::Mat::zeros(2, 400, CV_8UC1);
	cv::Rect rect;
	rect.x = 0;
	rect.y = 0;
	rect.height = 1;
	rect.width = 400;
	cv::Mat oneDate = handWrite01.reshape(1, 1);
	cv::Mat twoData = handWrite02.reshape(1, 1);
	oneDate.copyTo(testdata(rect));
	rect.y = 1;
	twoData.copyTo(testdata(rect));
	// 数据类型转换
	testdata.convertTo(testdata, CV_32F);

	// 进行估计识别
	cv::Mat result4;
	knn->findNearest(testdata, 5, result4);

	// 查看预测的结果
	for (int i = 0; i< result4.rows; i++)
	{
		int predict = result4.at<float>(i, 0);
		std::cout << "第" << i + 1 << "图像预测结果：" << predict 
			<< "  真实结果：" << i + 1 << std::endl;
	}
	std::cout << std::endl;
	cv::waitKey(0);
	cv::destroyAllWindows();

	// 加载测试数据
	cv::Mat data = cv::imread("../Img/textimages.png", cv::IMREAD_ANYDEPTH);
	cv::Mat labels = cv::imread("../Img/textlabels.png", cv::IMREAD_ANYDEPTH);
	data.convertTo(data, CV_32FC1);
	labels.convertTo(labels, CV_32SC1);
	cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(data, cv::ml::ROW_SAMPLE, labels);

	// 创建决策树(模型如何生成的,详情请看这块代码)
	cv::Ptr<cv::ml::DTrees> DTmodel = cv::ml::DTrees::create();
	// 参数设置
	// 下面两个参数是必需的
	DTmodel->setMaxDepth(8);  // 树的最大深度,输入参数为正整数
	DTmodel->setCVFolds(0);  // 交叉验证次数,一般使用0作为输入参数
	// 下面四个参数可以缺省，但是缺省会降低一定的精度
	// RTmodel->setUseSurrogates(false);  // 是否建立替代分裂点,输入参数为bool类型
	// RTmodel->setMinSampleCount(2);  	// 节点最小样本数量,当样本数量小于这个数值时,不再进行细分,输入参数为正整数
	// RTmodel->setUse1SERule(false);  // 是否严格修剪,剪枝即停止分支,输入参数为bool类型
	// RTmodel->setTruncatePrunedTree(false);  // 分支是否完全移除,输入参数为bool类型
	
	DTmodel->train(trainData);
	DTmodel->save("../Data/DTrees_model.yml");

	// 利用原数据进行测试
	cv::Mat result5;
	DTmodel->predict(data, result5);
	int count5 = 0;
	for (int row = 0; row < result5.rows; row++)
	{
		int predict = result5.at<float>(row, 0);
		if (labels.at<int>(row, 0) == predict)
		{
			count5 = count5 + 1;
		}
	}
	float rate5 = 1.0 * count5 / result5.rows;
	std::cout << "分类的正确性：" << rate5 << std::endl;
	std::cout << std::endl;
	cv::waitKey(0);


	// 构建随机森林RTrees类型变量
	cv::Ptr<cv::ml::RTrees> RTmodel = cv::ml::RTrees::create();
	// 参数设置
	// 设置迭代停止条件
	RTmodel->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100, 0.01));
	// 下列参数可以缺省以加快运行速度，但是会影响准确性
	// RTmodel->setUseSurrogates(false);  // 是否建立替代分裂点,输入参数为bool类型
	// RTmodel->setMinSampleCount(10);  // 节点最小样本数量,当样本数量小于这个数值时,不再进行细分,输入参数为正整数
	// RTmodel->setMaxDepth(10);  // 树的最大深度,输入参数为正整数
	// RTmodel->setRegressionAccuracy(0);  // 回归算法精度,输入参数为float类型
	// RTmodel->setMaxCategories(15);  // 最大类别数,输入参数为正整数
	// RTmodel->setPriors(Mat());  // 数据类型,输入值常为Mat()
	// RTmodel->setCalculateVarImportance(true);  // 是否需要计算Var,输入参数为bool类型
	// RTmodel->setActiveVarCount(4);  // 设置Var的数目,输出参数为正整数
	
	RTmodel->train(trainData);
	RTmodel->save("../Data/RTrees_model.yml");

	// 利用原数据进行测试
	cv::Mat result6;
	RTmodel->predict(data, result6);
	int count6 = 0;
	for (int row = 0; row < result6.rows; row++)
	{
		int predict = result6.at<float>(row, 0);
		if (labels.at<int>(row, 0) == predict)
		{
			count6 = count6 + 1;
		}
	}
	float rate = 1.0* count6 / result6.rows;
	std::cout << "分类的正确性：" << rate << std::endl;
	std::cout << std::endl;
	cv::waitKey(0);

	// 训练数据
	cv::Mat samples7, labls7;
	cv::FileStorage fread("../Data/point.yml", cv::FileStorage::READ);
	fread["data"] >> samples7;
	fread["labls"] >> labls7;
	fread.release();

	// 不同种类坐标点拥有不同的颜色
	std::vector<cv::Vec3b> colors7;
	colors7.push_back(cv::Vec3b(0, 255, 0));
	colors7.push_back(cv::Vec3b(0, 0, 255));

	// 创建空白图像用于显示坐标点
	cv::Mat img71(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat img72;
	img71.copyTo(img72);

	// 在空白图像中绘制坐标点
	for (int i = 0; i < samples7.rows; i++)
	{
		cv::Point2f point;
		point.x = samples7.at<float>(i, 0);
		point.y = samples7.at<float>(i, 1);
		cv::Scalar color = colors7[labls7.at<int>(i, 0)];
		cv::circle(img71, point, 3, color, -1);
		cv::circle(img72, point, 3, color, -1);
	}
	cv::imshow("两类像素点图像", img71);

	// 建立模型
	cv::Ptr<cv::ml::SVM> model = cv::ml::SVM::create();
	// 参数设置
	model->setKernel(cv::ml::SVM::INTER);  // 内核的模型
	model->setType(cv::ml::SVM::C_SVC);  // SVM的类型
	model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100, 0.01));	// 停止迭代的条件
	// model->setGamma(5.383);	// 设置算法中的gamma变量,默认值为1
	// model->setC(0.01);	// 设置算法中的c变量,默认值为0
	// model->setP(0.01);	// 设置算法中的epsilon变量,默认值为0
	// model->setNu(0.01);	// 设置算法中的nu变量,默认值为0
	// model->setDegree(3);	// 设置核函数中的读书,默认值为0
	
	// 训练模型
	model->train(cv::ml::TrainData::create(samples7, cv::ml::ROW_SAMPLE, labls7));

	// 用模型对图像中全部像素点进行分类
	cv::Mat imagePoint(1, 2, CV_32FC1);
	for (int y = 0; y < img72.rows; y = y + 2)
	{
		for (int x = 0; x < img72.cols; x = x + 2)
		{
			imagePoint.at<float>(0) = (float)x;
			imagePoint.at<float>(1) = (float)y;
			int color = (int)model->predict(imagePoint);
			img72.at<cv::Vec3b>(y, x) = colors7[color];
		}
	}

	cv::imshow("图像所有像素点分类结果", img72);
	cv::waitKey();
	/* OpenCV与深度神经网络应用实例 */
	/*
	OpenCV中的深度学习应用实例
	1. 图像分类
	2. 目标检测
	3. 人脸识别
	*/
	/*
	加载已有深度神经网络模型
	cv::dnn::readNet(	const String& model, 			// 模型文件
						const String& config = "", 		// 配置文件
						const String& framework = "" 	// 框架
						);								// 返回值cv::dnn::Net数据类型，表示神经网络模型
	
	向深度神经网络模型中输入数据
	cv::dnn::Net::setInput(	InputArray blob, 				// 输入数据
							const String& name = "" 		// 名称
							double scalefactor = 1.0, 		// 可选的标准化比例
							const Scalar& mean = Scalar()	// 可选的减数数值
							);

	转换输入到深度神经网络模型中的图像尺寸
	cv::dnn::blobFromImages(InputArray images, 				// 输入图像
							double scalefactor = 1.0, 		// 缩放因子
							const Size& size = Size(), 		// 输出图像的尺寸
							const Scalar& mean = Scalar(), 	// 像素值去均值化的数值
							bool swapRB = false, 			// 是否交换三通道图像的第一个通道和最后一个通道
							bool crop = false,				// 调整尺寸后是否对图像进行剪切的标志
							int deepth =CV_32F				// 输出图像的数据类型
							);								// 返回值cv::Mat数据类型，表示图像数据
	*/
	std::string model8 = "../Data/bvlc_googlenet.caffemodel";
	std::string config8 = "../Data/bvlc_googlenet.prototxt";

	// 加载模型
	cv::dnn::Net net1 = cv::dnn::readNet(model8, config8);
	if (net1.empty())
	{
		std::cout << "请确认是否输入空的模型文件" << std::endl;
		return -1;
	}

	// 获取各层信息
	std::vector<std::string> layerNames1 = net1.getLayerNames();
	for (int i = 0; i < layerNames1.size(); i++)
	{
		// 读取每层网络的ID
		int ID = net1.getLayerId(layerNames1[i]);
		// 读取每层网络的信息
		cv::Ptr<cv::dnn::Layer> layer = net1.getLayer(ID);
		// 输出网络信息
		std::cout << "网络层数：" << ID << "  网络层名称：" << layerNames1[i] << std::endl
			<< "网络层类型：" << layer->type.c_str() << std::endl;
	}

	cv::Mat faces = cv::imread("../Img/faces.jpg");
	if (faces.empty())
	{
		std::cout << "请确定是否输入正确的图像文件" << std::endl;
		return -1;
	}

	//读取人脸识别模型
	std::string model_bin2 = "../Data/ch12_face_age/opencv_face_detector_uint8.pb";
	std::string config_text2 = "../Data/ch12_face_age/opencv_face_detector.pbtxt";
	cv::dnn::Net faceNet = cv::dnn::readNet(model_bin2, config_text2);

	//读取性别检测模型
 	std::string genderProto = "../Data/ch12_face_age/gender_deploy.prototxt";
	std::string genderModel = "../Data/ch12_face_age/gender_net.caffemodel";
	std::string genderList[] = { "Male", "Female" };
	cv::dnn::Net genderNet = cv::dnn::readNet(genderModel, genderProto);
	if(faceNet.empty()&&genderNet.empty())
	{
		std::cout << "请确定是否输入正确的模型文件" << std::endl;
		return -1;
	}
	
	//对整幅图像进行人脸检测
	cv::Mat blobImage = cv::dnn::blobFromImage(faces, 1.0, cv::Size(300, 300), cv::Scalar(), false, false);
	faceNet.setInput(blobImage, "data");
	cv::Mat detect = faceNet.forward("detection_out");
	//人脸概率、人脸矩形区域的位置
	cv::Mat detectionMat(detect.size[2], detect.size[3], CV_32F, detect.ptr<float>());
	
	//对每个人脸区域进行性别检测
	int exBoundray = 25;  //每个人脸区域四个方向扩充的尺寸
	float confidenceThreshold = 0.5;  //判定为人脸的概率阈值，阈值越大准确性越高
	for (int i = 0; i < detectionMat.rows; i++) 
	{
		float confidence = detectionMat.at<float>(i, 2);  //检测为人脸的概率
		//只检测概率大于阈值区域的性别
		if (confidence > confidenceThreshold)
		{
			//网络检测人脸区域大小
			int topLx = detectionMat.at<float>(i, 3) * faces.cols;
			int topLy = detectionMat.at<float>(i, 4) * faces.rows;
			int bottomRx = detectionMat.at<float>(i, 5) * faces.cols;
			int bottomRy = detectionMat.at<float>(i, 6) * faces.rows;
			cv::Rect faceRect(topLx, topLy, bottomRx - topLx, bottomRy - topLy);

			//将网络检测出的区域尺寸进行扩充，要注意防止尺寸在图像真实尺寸之外
			cv::Rect faceTextRect;
			faceTextRect.x = cv::max(0, faceRect.x - exBoundray);
			faceTextRect.y = cv::max(0, faceRect.y - exBoundray);
			faceTextRect.width = cv::min(faceRect.width + exBoundray, faces.cols - 1);
			faceTextRect.height = cv::min(faceRect.height + exBoundray, faces.rows - 1);
			cv::Mat face = faces(faceTextRect);  //扩充后的人脸图像

			//调整面部图像尺寸
			cv::Mat faceblob = cv::dnn::blobFromImage(face, 1.0, cv::Size(227, 227), cv::Scalar(), false, false);
			//将调整后的面部图像输入到性别检测网络
			genderNet.setInput(faceblob);
			//计算检测结果
			cv::Mat genderPreds = genderNet.forward();  //两个性别的可能性

			//性别检测结果
			float male, female;
			male = genderPreds.at<float>(0, 0);
			female = genderPreds.at<float>(0, 1);
			int classID = male > female ? 0 : 1;
			std::string gender = genderList[classID];

			//在原图像中绘制面部轮廓和性别
			cv::rectangle(faces, faceRect, cv::Scalar(0, 0, 255), 2, 8, 0);
			cv::putText(faces, gender.c_str(), faceRect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, 8);
		}
	}
	cv::imshow("性别检测结果", faces);
	cv::waitKey(0);
	return 0;
}