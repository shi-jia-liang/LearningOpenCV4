#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
	cv::Mat img = cv::imread("../Img/Lena.jpg");
	if (img.empty())
	{
		std::cout << "图片不存在！" << std::endl;
		return -1;
	}

	/* 图像直方图的绘制 */
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	/*
	图像直方图的统计函数
	calcHist( 	const Mat* images,			输入图像
				int nimages,				输入图像的数量
				const int* channels,		需要统计的通道索引数组
				InputArray mask,			掩膜
				OutputArray hist_gray,			输出的统计直方图数据
				int dims,					需要计算直方图的维数
				const int* histSize,		存放每个维度直方图的数组的尺寸
				const float** ranges,		每个图象通道中灰度值的取值范围
				bool uniform = true,		直方图是否均匀的标志
				bool accumulate = false		是否累计统计直方图的标志(该参数主要用于统计多个图像整体的直方图)
				);
	*/
	cv::Mat hist_gray;			 // 用于存放直方图的计算结果
	const int channels[1] = {0}; // 通道索引
	float inRanges[2] = {0, 255};
	const float *ranges[1] = {inRanges}; // 像素灰度值范围
	const int bins[1] = {256};			 // 直方图的维度，其实就是像素灰度值的最大值

	cv::calcHist(&gray, 1, channels, cv::Mat(), hist_gray, 1, bins, ranges); // 计算图像直方图

	int hist_w = 512;
	int hist_h = 512;
	int width = 2;

	cv::Mat HistImage = cv::Mat::zeros(hist_w, hist_h, CV_8UC3);
	for (int i = 1; i <= hist_gray.rows; i++)
		cv::rectangle(HistImage, cv::Point(width * (i - 1), hist_h - 1), cv::Point(width * i - 1, hist_h - cvRound(hist_gray.at<float>(i - 1)) / 10), cv::Scalar(255, 255, 255), -1);

	cv::imshow("gray", gray);
	cv::imshow("HistImage", HistImage);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/* 直方图操作 */
	/*
	直方图归一化
	normalize( 	InputArray src, 				输入矩阵
				InputOutputArray dst, 			输入与src相同的矩阵，同时作为输出，输出结果是CV_32F类型的矩阵
				double alpha = 1, 				在范围归一化的情况下，归一化到下限边界的标准值
				double beta = 0,				范围归一化时的上限范围
				int norm_type = NORM_L2, 		归一化过程中数据范数种类的标志
				int dtype = -1, 				输出数据类型选择的标志
				InputArray mask = noArray()		掩膜
				);
	*/
	std::vector<double> positiveData = {2.0, 8.0, 10.0};
	std::vector<double> normalize_L1, normalize_L2, normalize_Inf, normalize_L2SQR;
	cv::normalize(positiveData, normalize_L1, 1, 0, cv::NORM_L1);	// 绝对值求和归一化
	cv::normalize(positiveData, normalize_L2, 1, 0, cv::NORM_L2);	// 模长归一化
	cv::normalize(positiveData, normalize_Inf, 1, 0, cv::NORM_INF); // 最大值归一化
	std::cout << "normalize_L1:[" << normalize_L1[0] << ", " << normalize_L1[1] << ", " << normalize_L1[2] << "]" << std::endl;
	std::cout << "normalize_L2:[" << normalize_L2[0] << ", " << normalize_L2[1] << ", " << normalize_L2[2] << "]" << std::endl;
	std::cout << "normalize_Inf:[" << normalize_Inf[0] << ", " << normalize_Inf[1] << ", " << normalize_Inf[2] << "]" << std::endl;

	cv::Mat hist_L1, hist_Inf;
	cv::Mat HistImage_L1 = cv::Mat::zeros(hist_w, hist_h, CV_8UC3);
	cv::Mat HistImage_Inf = cv::Mat::zeros(hist_w, hist_h, CV_8UC3);
	cv::normalize(hist_gray, hist_L1, 1, 0, cv::NORM_L1, -1, cv::Mat());
	for (int i = 1; i <= hist_L1.rows; i++)
		cv::rectangle(HistImage_L1, cv::Point(width * (i - 1), hist_h - 1), cv::Point(width * i - 1, hist_h - cvRound(30 * hist_h * hist_L1.at<float>(i - 1)) - 1), cv::Scalar(255, 0, 0), -1);

	cv::normalize(hist_gray, hist_Inf, 1, 0, cv::NORM_INF, -1, cv::Mat());
	for (int i = 1; i <= hist_Inf.rows; i++)
		cv::rectangle(HistImage_Inf, cv::Point(width * (i - 1), hist_h - 1), cv::Point(width * i - 1, hist_h - cvRound(hist_h * hist_Inf.at<float>(i - 1)) - 1), cv::Scalar(0, 0, 255), -1);

	cv::imshow("HistImage", HistImage);
	cv::imshow("HistImage_L1", HistImage_L1);
	cv::imshow("HistImage_Inf", HistImage_Inf);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// 直方图比较
	/*
	直方图比较
	compareHist(InputArray H1, 	比较图像的直方图1
				InputArray H2, 	比较图像的直方图2
				int method 		比较方法的标志
				);				返回值double,得到图像的相似性
	*/
	cv::Mat img_resize, img_pyrdown;
	cv::Mat hist_resize, hist_pyrdown;
	double resize_value, pyrdown_value, resizeANDpyrdown;
	cv::resize(gray, img_resize, cv::Size(), 0.5, 0.5); // 缩放图像
	cv::pyrDown(gray, img_pyrdown);						// 下采样图像

	cv::calcHist(&img_resize, 1, channels, cv::Mat(), hist_resize, 1, bins, ranges);   // 计算缩放图像直方图
	cv::calcHist(&img_pyrdown, 1, channels, cv::Mat(), hist_pyrdown, 1, bins, ranges); // 计算下采样图像直方图

	resize_value = cv::compareHist(hist_gray, hist_resize, cv::HISTCMP_CORREL);		   // 原图的直方图与缩放图像的直方图，相关法比较
	pyrdown_value = cv::compareHist(hist_gray, hist_pyrdown, cv::HISTCMP_CORREL);	   // 原图的直方图与下采样图像的直方图，相关法比较
	resizeANDpyrdown = cv::compareHist(hist_resize, hist_pyrdown, cv::HISTCMP_CORREL); // 缩放图像的直方图与下采样图像的直方图，相关法比较

	std::cout << std::endl;
	std::cout << "resize_value: " << resize_value << std::endl;
	std::cout << "pyrdown_value: " << pyrdown_value << std::endl;
	std::cout << "resizeANDpyrdown: " << resizeANDpyrdown << std::endl;
	// 通过上述,可知图像直方图具有缩放不变性,且图像缩放与图像下采样不等同

	/* 直方图应用 */
	// 直方图均衡化
	/*
	图像均衡化
	equalizeHist(	InputArray src, 输入图像(数据类型为CV_8UC1)
					OutputArray dst 均衡化后的图像
					);
	*/
	cv::Mat gray_equal;
	cv::equalizeHist(gray, gray_equal); // 图像均衡化

	cv::Mat hist_gray_equal;
	cv::calcHist(&gray_equal, 1, channels, cv::Mat(), hist_gray_equal, 1, bins, ranges); // 计算图像均衡化后直方图

	cv::Mat HistImage_equal = cv::Mat::zeros(hist_w, hist_h, CV_8UC3);
	for (int i = 1; i <= hist_gray_equal.rows; i++)
		cv::rectangle(HistImage_equal, cv::Point(width * (i - 1), hist_h - 1), cv::Point(width * i - 1, hist_h - cvRound(hist_gray_equal.at<float>(i - 1)) / 10), cv::Scalar(255, 255, 255), -1);

	cv::imshow("HistImage", HistImage);
	cv::imshow("HistImage_equal", HistImage_equal);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// 直方图匹配(均衡化是自动匹配灰度值,匹配是手动设定匹配灰度值)
	// Opencv4中，没有提供直方图匹配的函数,需要自己根据算法实现图像直方图匹配

	// 直方图反向投影(不做演示,因为该方法是通过直方图提取图像特征纹理,通过此直方图寻找其他图像是否有相同图像特征纹理。但实际应用中,具有相似的直方图,其图像特征纹理不一定相同)

	/* 图像的模板匹配 */
	// 由于直方图反向投影,并不能很好地找到相似纹理。因此,可以直接通过比较图像像素的形式来搜索是否存在相同的内容
	// 模板匹配常用于在一幅图像中寻找特定内容的任务
	/*
	matchTemplate( 	InputArray image, 			待模板匹配的原始图像
					InputArray templ,			模板图像
					OutputArray result, 		模板匹配结果输出图像
					int method, 				模板匹配方法的标志
					InputArray mask = noArray() 掩膜
					);
	*/
	cv::imshow("Img", img);											  // 原图
	cv::Mat img_temp = img(cv::Range(200, 400), cv::Range(200, 400)); // 抓取图像的一部分
	cv::Mat match_result;											  // 图像模板匹配的结果
	cv::matchTemplate(img, img_temp, match_result, cv::TM_CCOEFF_NORMED);
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	// 寻找匹配结果中的最大值和最小值以及坐标位置
	cv::minMaxLoc(match_result, &minVal, &maxVal, &minLoc, &maxLoc);
	cv::rectangle(img, cv::Rect(maxLoc.x, maxLoc.y, img_temp.cols, img_temp.rows), cv::Scalar(255, 255, 255), 2);

	cv::imshow("img_temp", img_temp);		  // 模板图像
	cv::imshow("match_result", match_result); // 匹配结果
	cv::imshow("Img_1", img);				  // 在原图上画出预测框
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}
