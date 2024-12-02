#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
	/* 像素距离与连通域 */
	/*
	实现图像的距离变换1(用于统计图像中所有像素距离"0像素"的最小距离)
	cv::distanceTransform( 	InputArray src, 					输入图像，数据类型为CV_8UC1
							OutputArray dst,					输出图像，数据类型为CV_8UC1或者CV_32FC1
							OutputArray labels, 				二维的标签数组，数据类型为CV_32FC1
							int distanceType, 					计算两个像素之间距离方法的标志
							int maskSize,						距离变换掩码矩阵尺寸
							int labelType = DIST_LABEL_CCOMP 	要构建的标签数据的类型
							);
	实现图像的距离变换2
	cv::distanceTransform( 	InputArray src, 					输入图像，数据类型为CV_8UC1
							OutputArray dst,					输出图像，数据类型为CV_8UC1或者CV_32FC1
							int distanceType, 					计算两个像素之间距离方法的标志
							int maskSize, 						距离变换掩码矩阵尺寸
							int dstType=CV_32F					输出图像的数据类型，可以是CV_8U或者CV_32F
							);
	*/
	cv::Mat a = (cv::Mat_<uchar>(5, 5) << 
				 1, 1, 1, 1, 1,
				 1, 1, 1, 1, 1,
				 1, 1, 1, 1, 0,
				 1, 1, 1, 1, 1,
				 1, 1, 1, 1, 1);
	cv::Mat dist_L1, dist_L2, dist_C;
	cv::distanceTransform(a, dist_L1, cv::DIST_L1, 5, CV_8U);
	std::cout << "街区距离： " << std::endl
			  << dist_L1 << std::endl;
	cv::distanceTransform(a, dist_L2, cv::DIST_L2, 5, CV_8U);
	std::cout << "欧氏距离： " << std::endl
			  << dist_L2 << std::endl;
	cv::distanceTransform(a, dist_C, cv::DIST_C, 5, CV_8U);
	std::cout << "棋盘距离： " << std::endl
			  << dist_C << std::endl;

	cv::Mat rice = cv::imread("../Img/rice.png");
	if(rice.empty()){
		std::cout << "图片不存在" << std::endl;
		return -1;
	}
	cv::Mat riceB;
	//将图像转成二值图像，用于统计连通域
	cvtColor(rice, riceB, cv::COLOR_BGR2GRAY);
	cv::Mat riceBW, riceBW_INV;
	// 将图像装换成二值图像，同时把黑白区域颜色互换
	cv::threshold(riceB, riceBW, 40, 255, cv::THRESH_BINARY);
	cv::threshold(riceB, riceBW_INV, 40, 255, cv::THRESH_BINARY_INV);
	// 距离变换
	cv::Mat dist, dist_INV;
	cv::distanceTransform(riceBW, dist, cv::DIST_L1, 3);
	cv::distanceTransform(riceBW_INV, dist_INV, cv::DIST_L1, 3);
	// 转换为可视化结果（归一化）
    cv::Mat distNorm, dist_INVNorm;
    cv::normalize(dist, distNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(dist_INV, dist_INVNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
	// 显示结果
	cv::imshow("rice", rice);
	cv::imshow("riceBW", riceBW);
	cv::imshow("riceBW_INV", riceBW_INV);
	cv::imshow("dist", dist);
	cv::imshow("dist_INV", dist_INV);
	cv::imshow("distNorm", distNorm);
	cv::imshow("dist_INVNorm", dist_INVNorm);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// 提取图像连通域
	/*
	提取图像连通域1
	cv::connectedComponents(InputArray image, 		输入图像，数据类型为CV_8U
							OutputArray labels,		标记不同连通域后的输出图像
                            int connectivity, 		标记连通域时使用的领域种类，4表示4-领域，8表示8领域
							int ltype, 				输出图像的数据类型，目前支持CV_32S和CV_16U
							int ccltype 			标记连通域时使用的算法类型标志
							);						返回值int类型，表示图像中连通域的数目
	提取图像连通域2
	cv::connectedComponents(InputArray image, 		输入图像，数据类型为CV_8U
							OutputArray labels,		标记不同连通域后的输出图像
                            int connectivity = 8, 	标记连通域时使用的领域种类，4表示4-领域，8表示8领域
							int ltype = CV_32S		输出图像的数据类型，目前支持CV_32S和CV_16U
							);						返回值int类型，表示图像中连通域的数目
	*/
	//生成随机颜色，用于区分不同连通域
	cv::RNG rng(10086);
	cv::Mat connectout;
	int number = cv::connectedComponents(riceBW, connectout, 8, CV_16U);
	std::vector<cv::Vec3b> colors;
	for (int i = 0; i < number; i++)
	{
		//使用均匀分布的随机数确定颜色
		cv::Vec3b vec3 = cv::Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		colors.push_back(vec3);
	}

	//以不同颜色标记出不同的连通域
	cv::Mat result = cv::Mat::zeros(rice.size(), rice.type());
	int w1 = result.cols;
	int h1 = result.rows;
	for (int row = 0; row < h1; row++)
	{
		for (int col = 0; col < w1; col++)
		{
			int label = connectout.at<ushort>(row, col);
			if (label == 0)  //背景的黑色不改变
			{
				continue;
			}
			result.at<cv::Vec3b>(row, col) = colors[label];
		}
	}
	//显示结果
	cv::imshow("原图", rice);
	cv::imshow("标记后的图像", result);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/*
	更详细地提取图像连通域1(中心位置、矩形区域大小、区域面积等信息)
	cv::connectedComponentsWithStats(	InputArray image, 		输入图像，数据类型为CV_8U
										OutputArray labels,		标记不同连通域后的输出图像
      									OutputArray stats, 		含有不同连通域统计信息的矩阵，矩阵类型为CV_32S
										OutputArray centroids,	统计连通域质心的坐标
      									int connectivity, 		统计连通域时使用的领域种类
										int ltype, 				输出图像的数据类型，目前支持CV_32S和CV_16U
										int ccltype				标记连通域使用的算法类型标志
										);						返回值int类型，表示图像中连通域的数目
	更详细地提取图像连通域2
	cv::connectedComponentsWithStats(	InputArray image, 		输入图像，数据类型为CV_8U
										OutputArray labels,		标记不同连通域后的输出图像
                                    	OutputArray stats, 		含有不同连通域统计信息的矩阵，矩阵类型为CV_32S
										OutputArray centroids,	统计连通域质心的坐标
                                    	int connectivity = 8, 	统计连通域时使用的领域种类
										int ltype = CV_32S		输出图像的数据类型，目前支持CV_32S和CV_16U
										);						返回值int类型，表示图像中连通域的数目
	*/
	// 统计连通域的信息
	cv::Mat out, stats, centroids;
	// 统计图像中连通域的个数
	int number2 = connectedComponentsWithStats(riceBW, out, stats, centroids, 8, CV_16U);
	// 以不同颜色标记出不同的连通域
	cv::Mat result2 = cv::Mat::zeros(rice.size(), rice.type());
	int w2 = result2.cols;
	int h2 = result2.rows;
	for (int i = 1; i < number2; i++)
	{
		// 中心位置
		int center_x = centroids.at<double>(i, 0);
		int center_y = centroids.at<double>(i, 1);
		// 矩形边框
		int x = stats.at<int>(i, cv::CC_STAT_LEFT);
		int y = stats.at<int>(i, cv::CC_STAT_TOP);
		int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
		int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		// 中心位置绘制
		cv::circle(rice, cv::Point(center_x, center_y), 2, cv::Scalar(0, 255, 0), 2, 8, 0);
		// 外接矩形
		cv::Rect rect(x, y, w, h);
		cv::rectangle(rice, rect, colors[i], 1, 8, 0);
		cv::putText(rice, cv::format("%d", i), cv::Point(center_x, center_y),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
		std::cout << "number: " << i << ",area: " << area << std::endl;
	}
	// 显示结果
	cv::imshow("标记后的图像", rice);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/* 腐蚀和膨胀 */
	/* 形态学应用 */
	return 0;
}