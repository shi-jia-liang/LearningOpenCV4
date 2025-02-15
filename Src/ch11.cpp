#include <iostream>
#include <opencv2/opencv.hpp>
std::vector<cv::Scalar> color_lut;  //颜色查找表
void draw_lines(cv::Mat &image, std::vector<cv::Point2f> pt1, std::vector<cv::Point2f> pt2)
{
	cv::RNG rng(5000);
	if (color_lut.size() < pt1.size())
	{
		for (size_t t = 0; t < pt1.size(); t++)
		{
			color_lut.push_back(cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255)));
		}
	}
	for (size_t t = 0; t < pt1.size(); t++) {
		cv::line(image, pt1[t], pt2[t], color_lut[t], 2, 8, 0);
	}
}

int main(){
	/* 差值法检测移动物体 */
	/* 
	计算两幅图像的差值的绝对值
	cv::absdiff(src1, 		// 输入图像1
				src2, 		// 输入图像2
				dst 		// 输出图像
				);
	*/
	// 加载视频文件，并判断是否加载成功
	cv::VideoCapture bikeVideo("../Video/bike.avi");
	if (!bikeVideo.isOpened()) {
		std::cout << "请确认视频文件是否正确" << std::endl;
		return -1;
	}
	// 输出视频相关信息
	int fps1 = bikeVideo.get(cv::CAP_PROP_FPS);
	int width1 = bikeVideo.get(cv::CAP_PROP_FRAME_WIDTH);
	int height1 = bikeVideo.get(cv::CAP_PROP_FRAME_HEIGHT);
	int num_of_frames1 = bikeVideo.get(cv::CAP_PROP_FRAME_COUNT);
	std::cout << "视频宽度：" << width1 << std::endl <<  " 视频高度：" << height1 << std::endl << " 视频帧率：" << fps1 << std::endl << " 视频总帧数: " << num_of_frames1 << std::endl;

	// 读取视频中第一帧图像作为前一帧图像，并进行灰度化
	cv::Mat preFrame1, preGray1;
	bikeVideo.read(preFrame1);
	cv::cvtColor(preFrame1, preGray1, cv::COLOR_BGR2GRAY);
	// 对图像进行高斯滤波，减少噪声干扰
	cv::GaussianBlur(preGray1, preGray1, cv::Size(0, 0), 15);

	cv::Mat binary1;
	cv::Mat frame1, gray1;
	// 形态学操作的矩形模板
	cv::Mat k1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7), cv::Point(-1, -1));

	while (true) 
	{
		// 视频中所有图像处理完后推出循环
		if (!bikeVideo.read(frame1))
		{
			break;
		}

		// 对当前帧进行灰度化
		cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
		cv::GaussianBlur(gray1, gray1, cv::Size(0, 0), 15);

		// 计算当前帧与前一帧的差值的绝对值
		cv::absdiff(gray1, preGray1, binary1);
		
		// 对计算结果二值化并进行开运算，减少噪声的干扰
		cv::threshold(binary1, binary1, 10, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		cv::morphologyEx(binary1, binary1, cv::MORPH_OPEN, k1);

		// 显示处理结果
		cv::imshow("input", frame1);
		cv::imshow("result", binary1);
		
		// 将当前帧变成前一帧，准备下一个循环，注释掉这句话为固定背景
		gray1.copyTo(preGray1); // gray->preGray

		// 5毫秒延时判断是否推出程序，按ESC键退出
		char c = cv::waitKey(5);
		if (c == 27) 
		{
			break;
		}
	}
	//释放资源
	bikeVideo.release();
	cv::destroyAllWindows();

	/* 均值迁移法目标跟踪 */
	/*
	均值迁移法是一种基于密度估计的非参数聚类算法
	均值迁移法的目标跟踪步骤：
	1. 初始化目标区域
	2. 计算目标区域的直方图
	3. 计算目标区域的质心
	4. 计算目标区域的直方图的质心
	5. 计算目标区域的质心与直方图的质心的距离
	6. 如果距离小于阈值，则认为目标区域没有发生变化，否则，更新目标区域
	*/
	/*
	均值迁移法的目标跟踪函数
	cv::meanShift(	src, 		// 输入图像
				  	dst, 		// 输出图像
				  	window, 	// 目标区域
				  	criteria 	// 停止条件
				  	);
	
	通过鼠标在图像中选择目标区域
	cv::selectROI(	windowName, 	// 窗口名称
					src, 			// 输入图像
					showCrossair, 	// 是否显示十字
					fromCenter 		// 是否从中心开始选择
					);
	
	自适应均值迁移法的目标跟踪函数				
	cv::CamShift(	src, 		// 输入图像
				  	window, 	// 目标区域
				  	criteria 	// 停止条件
				  	);
	*/
	//打开视频文件，并判断是否成功打开
	cv::VideoCapture vtest("../Video/vtest.avi");
	if (!vtest.isOpened())
	{
		std::cout << "请确认输入的视频文件名是否正确" << std::endl;
		return -1;
	}

	//是否已经计算目标区域直方图标志，0表示没有计算，1表示已经计算
	int trackObject2 = 0;  
	
	//计算直方图和反向直方图相关参数
	int hsize2 = 16;
	float hranges2[] = { 0,180 };
	const float* phranges2 = hranges2;
	
	//选择目标区域
	cv::Mat frame2, hsv2, hue2, hist2, histimg2 = cv::Mat::zeros(200, 320, CV_8UC3), backproj2;
	vtest.read(frame2);
	cv::Rect selection2 = cv::selectROI("选择目标跟踪区域", frame2, true, false);

	while (true)
	{
		//判断是否读取了全部图像
		if (!vtest.read(frame2))
		{
			break;
		}
		//将图像转化成HSV颜色空间
		cvtColor(frame2, hsv2, cv::COLOR_BGR2HSV);

		//定义计算直方图和反向直方图相关数据和图像
		int ch[] = { 0, 0 };
		hue2.create(hsv2.size(), hsv2.depth());
		mixChannels(&hsv2, 1, &hue2, 1, ch, 1);

		//是否已经完成跟踪目标直方图的计算
		if (trackObject2 <= 0)
		{
			//目标区域的HSV颜色空间
			cv::Mat roi(hue2, selection2);
			//计算直方图和直方图归一化
			calcHist(&roi, 1, 0, roi, hist2, 1, &hsize2, &phranges2);
			normalize(hist2, hist2, 0, 255, cv::NORM_MINMAX);
			
			//将标志设置为1，不再计算目标区域的直方图
			trackObject2 = 1; // Don't set up again, unless user selects new ROI

			//显示目标区域的直方图，可以将注释掉，不影响跟踪效果
			histimg2 = cv::Scalar::all(0);
			int binW = histimg2.cols / hsize2;
			cv::Mat buf(1, hsize2, CV_8UC3);
			for (int i = 0; i < hsize2; i++)
				buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180. / hsize2), 255, 255);
			cvtColor(buf, buf, cv::COLOR_HSV2BGR);

			for (int i = 0; i < hsize2; i++)
			{
				int val = cv::saturate_cast<int>(hist2.at<float>(i)*histimg2.rows / 255);
				cv::rectangle(histimg2, cv::Point(i*binW, histimg2.rows),
					cv::Point((i + 1)*binW, histimg2.rows - val),
					cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8);
			}
		}

		// 计算目标区域的方向直方图
		cv::calcBackProject(&hue2, 1, 0, hist2, backproj2, &phranges2);
		
		//均值迁移法跟踪目标
		cv::meanShift(backproj2, selection2, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
		//再图像中绘制寻找到的跟踪窗口
		cv::rectangle(frame2, selection2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);

		//显示结果
		cv::imshow("CamShift Demo", frame2);  //显示跟踪结果
		cv::imshow("Histogram", histimg2);  //显示目标区域直方图

		//按ESC键退出程序
		char c = (char)cv::waitKey(50);
		if (c == 27)
			break;
	}
	//释放资源
	vtest.release();
	cv::destroyAllWindows();

	cv::VideoCapture mulballs("../Video/mulballs.mp4");
	if (!mulballs.isOpened())
	{
		std::cout << "请确认输入的视频文件名是否正确" << std::endl;
		return -1;
	}

	//是否已经计算目标区域直方图标志，0表示没有计算，1表示已经计算
	int trackObject3 = 0;

	//计算直方图和反向直方图相关参数
	int hsize3 = 16;
	float hranges3[] = { 0,180 };
	const float* phranges3 = hranges3;

	//选择目标区域
	cv::Mat frame3, hsv3, hue3, hist3, histImg3 = cv::Mat::zeros(200, 320, CV_8UC3), backproj3;
	mulballs.read(frame3);
	cv::Rect selection3 = cv::selectROI("选择目标跟踪区域", frame3, true, false);
	cv::Rect selection_Cam = selection3;
	while (true)
	{
		//判断是否读取了全部图像
		if (!mulballs.read(frame3))
		{
			break;
		}
		//将图像转化成HSV颜色空间
		cv::cvtColor(frame3, hsv3, cv::COLOR_BGR2HSV);

		//定义计算直方图和反向直方图相关数据和图像
		int ch[] = { 0, 0 };
		hue3.create(hsv3.size(), hsv3.depth());
		cv::mixChannels(&hsv3, 1, &hue3, 1, ch, 1);

		//是否已经完成跟踪目标直方图的计算
		if (trackObject3 <= 0)
		{
			//目标区域的HSV颜色空间
			cv::Mat roi(hue3, selection3);
			//计算直方图和直方图归一化
			cv::calcHist(&roi, 1, 0, roi, hist3, 1, &hsize3, &phranges3);
			cv::normalize(hist3, hist3, 0, 255, cv::NORM_MINMAX);

			//将标志设置为1，不再计算目标区域的直方图
			trackObject3 = 1; // Don't set up again, unless user selects new ROI

			//显示目标区域的直方图，可以将注释掉，不影响跟踪效果
			int binW = histImg3.cols / hsize3;
			cv::Mat b(1, hsize3, CV_8UC3);
			for (int i = 0; i < hsize3; i++)
				b.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180. / hsize3), 255, 255);
			cvtColor(b, b, cv::COLOR_HSV2BGR);
			for (int i = 0; i < hsize3; i++)
			{
				int val = cv::saturate_cast<int>(hist3.at<float>(i)*histImg3.rows / 255);
				cv::rectangle(histImg3, cv::Point(i*binW, histImg3.rows),	cv::Point((i + 1)*binW, histImg3.rows - val),
				cv::Scalar(b.at<cv::Vec3b>(i)), -1, 8);
			}
		}

		// 计算目标区域的方向直方图
		cv::calcBackProject(&hue3, 1, 0, hist3, backproj3, &phranges3);

		cv::Mat frame_Cam;
		frame3.copyTo(frame_Cam);

		//均值迁移法跟踪目标
		cv::meanShift(backproj3, selection3,cv:: TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
		//在图像中绘制寻找到的跟踪窗口
		cv::rectangle(frame3, selection3, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);

		//自适应均值迁移法跟踪目标
		cv::RotatedRect trackBox = CamShift(backproj3, selection_Cam,
			cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
		//绘制椭圆窗口
		cv::ellipse(frame_Cam, trackBox, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);

		//显示结果
		cv::imshow("meanShift跟踪结果", frame3);  //显示跟踪结果
		cv::imshow("CamShift跟踪结果", frame_Cam);  //显示跟踪结果
		cv::imshow("Histogram", histImg3);  //显示目标区域直方图

		//按ESC键退出程序
		char c = (char)cv::waitKey(50);
		if (c == 27)
			break;
	}
	//释放资源
	mulballs.release();
	cv::destroyAllWindows();
	
	/* 光流法目标跟踪 */
	/*
	光流法是一种基于图像亮度的运动估计方法
	光流法的目标跟踪步骤：
	1. 初始化目标区域
	2. 计算目标区域的光流
	3. 计算目标区域的质心
	4. 计算目标区域的光流的质心
	5. 计算目标区域的质心与光流的质心的距离
	6. 如果距离小于阈值，则认为目标区域没有发生变化，否则，更新目标区域
	*/
	/*
	Faeneback多项式扩展算法光流法的目标跟踪函数
	cv::calcOpticalFlowFarneback(	prevImg, 	// 前一帧图像
									nextImg, 	// 当前帧图像
									flow, 		// 光流
									0.5, 		// 图像金字塔尺度因子
									3, 			// 图像金字塔层数
									15, 		// 窗口大小
									3, 			// 多项式扩展系数
									5, 			// 迭代次数
									1.1, 		// 高斯标准差
									0 			// 光流法的操作
									);
	
	
	计算二维向量的模长与方向
	cv::cartToPolar(	x, 	// x方向
						y, 	// y方向
						mag, 	// 模长
						angle 	// 方向
						);
	
	LK稀疏光流法的目标跟踪函数
	cv::calcOpticalFlowPyrLK(	prevImg, 	// 前一帧图像
								nextImg, 	// 当前帧图像
								prevPts, 	// 前一帧图像的特征点
								nextPts, 	// 当前帧图像的特征点
								status, 	// 特征点是否找到
								err, 		// 特征点的误差
								cv::Size(15, 15), 	// 搜索窗口大小
								5, 		// 金字塔层数
								cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03), 	// 停止条件
								cv::OPTFLOW_USE_INITIAL_FLOW 	// 使用初始光流
								);
	*/
	cv::VideoCapture vtest2("../Video/vtest.avi");
	cv::Mat prevFrame4, prevGray4;
	if (!vtest2.read(prevFrame4))
	{
		std::cout << "请确认视频文件名称是否正确" << std::endl;
		return -1;
	}

	//将彩色图像转换成灰度图像
	cv::cvtColor(prevFrame4, prevGray4, cv::COLOR_BGR2GRAY);
	
	while (true) 
	{	
		cv::Mat nextFrame, nextGray;
		//所有图像处理完成后推出程序
		if (!vtest2.read(nextFrame))
		{
			break;
		}
		cv::imshow("视频图像", nextFrame);

		//计算稠密光流
		cv::cvtColor(nextFrame, nextGray, cv::COLOR_BGR2GRAY);
		cv::Mat_<cv::Point2f> flow;  //两个方向的运动速度
		cv::calcOpticalFlowFarneback(prevGray4, nextGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
	
		cv::Mat xV = cv::Mat::zeros(prevFrame4.size(), CV_32FC1);  //x方向移动速度
		cv::Mat yV = cv::Mat::zeros(prevFrame4.size(), CV_32FC1);  //y方向移动速度
		//提取两个方向的速度
		for (int row = 0; row < flow.rows; row++)
		{
			for (int col = 0; col < flow.cols; col++)
			{
				const cv::Point2f& flow_xy = flow.at<cv::Point2f>(row, col);
				xV.at<float>(row, col) = flow_xy.x;
				yV.at<float>(row, col) = flow_xy.y;
			}
		}
		
		//计算向量角度和幅值
		cv::Mat magnitude4, angle4;
		cv::cartToPolar(xV, yV, magnitude4, angle4);

		//讲角度转换成角度制
		angle4 = angle4 * 180.0 / CV_PI / 2.0;

		//把幅值归一化到0-255区间便于显示结果
		cv::normalize(magnitude4, magnitude4, 0, 255, cv::NORM_MINMAX);

		//计算角度和幅值的绝对值
		cv::convertScaleAbs(magnitude4, magnitude4);
		cv::convertScaleAbs(angle4, angle4);

		//讲运动的幅值和角度生成HSV颜色空间的图像
		cv::Mat HSV = cv::Mat::zeros(prevFrame4.size(), prevFrame4.type());
		std::vector<cv::Mat> result;
		cv::split(HSV, result);
		result[0] = angle4;  //决定颜色
		result[1] = cv::Scalar(255);
		result[2] = magnitude4;  //决定形态
		//将三个多通道图像合并成三通道图像
		cv::merge(result, HSV);
		
		//讲HSV颜色空间图像转换到RGB颜色空间中
		cv::Mat rgbImg;
		cv::cvtColor(HSV, rgbImg, cv::COLOR_HSV2BGR);
		
		//显示检测结果
		cv::imshow("运动检测结果", rgbImg);
		int ch = cv::waitKey(5);
		if (ch == 27) 
		{
			break;
		}
	}
	//释放资源
	vtest2.release();
	cv::waitKey(0);
	cv::destroyAllWindows();

	cv::VideoCapture capture("../Video/mulballs.mp4");
	cv::Mat prevframe5, prevImg5;
	if (!capture.read(prevframe5))
	{
		std::cout << "请确认输入视频文件是否正确" << std::endl;
		return -1;
	}
	cv::cvtColor(prevframe5, prevImg5, cv::COLOR_BGR2GRAY);

	//角点检测相关参数设置
	std::vector<cv::Point2f> Points;
	double qualityLevel = 0.01;
	int minDistance = 10;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k5 = 0.04;
	int Corners = 5000;

	//角点检测
	cv::goodFeaturesToTrack(prevImg5, Points, Corners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k5);

	//稀疏光流检测相关参数设置
	std::vector<cv::Point2f> prevPts;  //前一帧图像角点坐标
	std::vector<cv::Point2f> nextPts;  //当前帧图像角点坐标
	std::vector<uchar> status;  //检点检测到的状态
	std::vector<float> err;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT
		+ cv::TermCriteria::EPS, 30, 0.01);
	double derivlambda = 0.5;
	int flags = 0;

	//初始状态的角点
	std::vector<cv::Point2f> initPoints;
	initPoints.insert(initPoints.end(), Points.begin(), Points.end());

	//前一帧图像中的角点坐标
	prevPts.insert(prevPts.end(), Points.begin(), Points.end());

	while (true)
	{
		cv::Mat nextframe, nextImg;
		if (!capture.read(nextframe))
		{
			break;
		}
		cv::imshow("nextframe", nextframe);

		//光流跟踪
		cv::cvtColor(nextframe, nextImg, cv::COLOR_BGR2GRAY);
		cv::calcOpticalFlowPyrLK(prevImg5, nextImg, prevPts, nextPts, status, err,
			cv::Size(31, 31), 3, criteria, derivlambda, flags);

		//判断角点是否移动，如果不移动就删除
		size_t i, k;
		for (i = k = 0; i < nextPts.size(); i++)
		{
			// 距离与状态测量
			double dist = abs(prevPts[i].x - nextPts[i].x) + abs(prevPts[i].y - nextPts[i].y);
			if (status[i] && dist > 2)
			{
				prevPts[k] = prevPts[i];
				initPoints[k] = initPoints[i];
				nextPts[k++] = nextPts[i];
				cv::circle(nextframe, nextPts[i], 3, cv::Scalar(0, 255, 0), -1, 8);
			}
		}

		//更新移动角点数目
		nextPts.resize(k);
		prevPts.resize(k);
		initPoints.resize(k);

		// 绘制跟踪轨迹
		draw_lines(nextframe, initPoints, nextPts);
		cv::imshow("result", nextframe);

		char c = cv::waitKey(50);
		if (c == 27)
		{
			break;
		}

		//更新角点坐标和前一帧图像
		std::swap(nextPts, prevPts);
		nextImg.copyTo(prevImg5);

		//如果角点数目少于30，就重新检测角点
		if (initPoints.size() < 30)
		{
			cv::goodFeaturesToTrack(prevImg5, Points, Corners, qualityLevel,
				minDistance, cv::Mat(), blockSize, useHarrisDetector, k);
			initPoints.insert(initPoints.end(), Points.begin(), Points.end());
			prevPts.insert(prevPts.end(), Points.begin(), Points.end());
			std::printf("total feature points : %d\n", prevPts.size());
		}

	}

	return 0;
}