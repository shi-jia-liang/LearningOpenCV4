#include <iostream>
#include <opencv2/opencv.hpp>

int main(){
	/* 傅里叶变换 */
	// 图像可以看作是二维信号，高频区域体现的是图像的细节、纹理信息；低频区域体现的是图像的轮廓信息、光照信息
	/* 
	傅里叶变换函数
	cv::dft(InputArray src, 		// 输入图像
			OutputArray dst, 		// 输出图像
			int flags = 0, 			// 变换标志
			int nonzeroRows = 0 	// 非零行数
			);

	傅里叶逆变换函数
	cv::idft(	InputArray src, 		// 输入图像
				OutputArray dst, 		// 输出图像
				int flags = 0, 			// 变换标志
				int nonzeroRows = 0 	// 非零行数
				);
	
	傅里叶变换的优化尺寸函数(进行尺寸变化，使得计算更快)
	cv::getOptimalDFTSize(	int vecsize	// 输入图像的行数或者列数
							);			// 返回值int数据类型，返回最佳的傅里叶变换尺寸

	在图像周围形成外框函数					
	cv::copyMakeBorder(	InputArray src, 					// 输入图像
						OutputArray dst,					// 输出图像
                        int top, 							// 扩展上边界的像素行数
						int bottom, 						// 扩展下边界的像素行数
						int left, 							// 扩展左边界的像素列数
						int right,							// 扩展右边界的像素列数
                        int borderType, 					// 扩展边界类型
						const Scalar& value = Scalar() 		// 扩展边界时使用的数值
						);
	
	计算两个矩阵对应位置组成的向量的幅值函数
	cv::magnitude(	InputArray x, 			// 向量x坐标的浮点矩阵
					InputArray y, 			// 向量y坐标的浮点矩阵
					OutputArray magnitude	// 输出的幅值矩阵
					);

	计算两个复数矩阵的乘积函数
	cv::mulSpectrums(	InputArray a, 		// 输入矩阵a
						InputArray b, 		// 输入矩阵b
						OutputArray c,		// 输出矩阵c
                    	int flags, 			// 操作标志
						bool conjB = false	// 是否对矩阵b进行共轭操作的标志,默认为false,不进行共轭变换
						);

	离散余弦变换函数
	cv::dct(InputArray src, 	// 输入图像
			OutputArray dst, 	// 输出图像	
			int flags = 0 		// 变换标志
			);

	离散余弦逆变换函数
	cv::idct(	InputArray src, 	// 输入图像
				OutputArray dst, 	// 输出图像
				int flags = 0 		// 变换标志
				);
	*/
	cv::Mat lena = cv::imread("../Img/lena.jpg");
	cv::Mat lena_gray;
	if(lena.empty()){
		std::cout << "图片不存在" << std::endl;
		return -1; 
	}
	cv::cvtColor(lena, lena_gray, cv::COLOR_BGR2GRAY);
	cv::Mat grayfloat = cv::Mat_<float>(lena_gray);  //更改图像数据类型为float
	cv::resize(lena_gray, lena_gray, cv::Size(502, 502));
	cv::imshow("原图像", lena_gray);

	// 计算合适的离散傅里叶变换尺寸
	int rows = cv::getOptimalDFTSize(lena_gray.rows); 
	int cols = cv::getOptimalDFTSize(lena_gray.cols);

	// 扩展图像
	cv::Mat appropriate;
	int T = (rows - lena_gray.rows) / 2;  //上方扩展行数
	int B = rows - lena_gray.rows - T;  //下方扩展行数
	int L = (cols - lena_gray.cols) / 2;  //左侧扩展行数
	int R = cols - lena_gray.cols - L;  //右侧扩展行数
	cv::copyMakeBorder(lena_gray, appropriate, T, B, L, R, cv::BORDER_CONSTANT);
	cv::imshow("扩展后的图像", appropriate);

	// 构建离散傅里叶变换输入量
	cv::Mat flo[2], complex;
	flo[0] = cv::Mat_<float>(appropriate);  //实数部分
	flo[1] = cv::Mat::zeros(appropriate.size(), CV_32F);  //虚数部分
	cv::merge(flo, 2, complex);  //合成一个多通道矩阵

	// 进行离散傅里叶变换
	cv::Mat result1;
	cv::dft(complex, result1);

	// 将复数转化为幅值
	cv::Mat resultC[2];
	cv::split(result1, resultC);  //分成实数和虚数
	cv::Mat amplitude;
	cv::magnitude(resultC[0], resultC[1], amplitude);

	// 进行对数放缩公式为： M1 = log（1+M），保证所有数都大于0
	amplitude = amplitude + 1;
	cv::log(amplitude, amplitude);//求自然对数

	// 与原图像尺寸对应的区域								
	amplitude = amplitude(cv::Rect(T, L, lena_gray.cols, lena_gray.rows));
	cv::normalize(amplitude, amplitude, 0, 1, cv::NORM_MINMAX);  //归一化
	cv::imshow("傅里叶变换结果幅值图像", amplitude);  //显示结果

	// 重新排列傅里叶图像中的象限，使得原点位于图像中心
	int centerX = amplitude.cols / 2;
	int centerY = amplitude.rows / 2;
	// 分解成四个小区域
	cv::Mat Qlt(amplitude, cv::Rect(0, 0, centerX, centerY));//ROI区域的左上
	cv::Mat Qrt(amplitude, cv::Rect(centerX, 0, centerX, centerY));//ROI区域的右上
	cv::Mat Qlb(amplitude, cv::Rect(0, centerY, centerX, centerY));//ROI区域的左下
	cv::Mat Qrb(amplitude, cv::Rect(centerX, centerY, centerX, centerY));//ROI区域的右下

	// 交换象限，左上和右下进行交换
	cv::Mat med;
	Qlt.copyTo(med);
	Qrb.copyTo(Qlt);
	med.copyTo(Qrb);
	// 交换象限，左下和右上进行交换
	Qrt.copyTo(med);
	Qlb.copyTo(Qrt);
	med.copyTo(Qlb);

	cv::imshow("中心化后的幅值图像", amplitude);

	// 构建滤波器
	cv::Mat kernel = (cv::Mat_<float>(5, 5) << 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1);
	// 构建输出图像
	cv::Mat result2;
	int rwidth = abs(grayfloat.rows - kernel.rows) + 1;
	int rheight = abs(grayfloat.cols - kernel.cols) + 1;
	result2.create(rwidth, rheight, grayfloat.type());

	// 计算最优离散傅里叶变换尺寸
	int width = cv::getOptimalDFTSize(grayfloat.cols + kernel.cols - 1);
	int height = cv::getOptimalDFTSize(grayfloat.rows + kernel.rows - 1);

	// 改变输入图像尺寸
	cv::Mat tempA;
	int A_T = 0;
	int A_B = width - grayfloat.rows;
	int A_L = 0;
	int A_R = height - grayfloat.cols;
	copyMakeBorder(grayfloat, tempA, 0, A_B, 0, A_R, cv::BORDER_CONSTANT);

	// 改变滤波器尺寸
	cv::Mat tempB;
	int B_T = 0;
	int B_B = width - kernel.rows;
	int B_L = 0;
	int B_R = height - kernel.cols;
	copyMakeBorder(kernel, tempB, 0, B_B, 0, B_R, cv::BORDER_CONSTANT);

	// 分别进行离散傅里叶变换
	cv::dft(tempA, tempA, 0, grayfloat.rows);
	cv::dft(tempB, tempB, 0, kernel.rows);

	// 多个傅里叶变换的结果相乘
	mulSpectrums(tempA, tempB, tempA, cv::DFT_COMPLEX_OUTPUT);

	// 相乘结果进行逆变换
	//dft(tempA, tempA, DFT_INVERSE | DFT_SCALE, result.rows);
	idft(tempA, tempA, cv::DFT_SCALE, result2.rows);

	//对逆变换结果进行归一化
	normalize(tempA, tempA, 0, 1, cv::NORM_MINMAX);

	//截取部分结果作为滤波结果 
	tempA(cv::Rect(0, 0, result2.cols, result2.rows)).copyTo(result2);

	//显示结果
	cv::imshow("原图像", lena_gray);
	cv::imshow("滤波结果", result2);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/* 积分图像 */
	/*
	积分图像函数(标准求和积分)
	cv::integral(	InputArray src, 		// 输入图像
					OutputArray sum, 		// 输出标准求和积分图像
					int sdepth = -1 		// 输出标准求和积分图像的数据类型标志,-1表示满足数据存储的自适应类型
					);

	积分图像函数(平方求和积分)
	cv::integral(	InputArray src, 		// 输入图像
					OutputArray sum, 		// 输出标准求和积分图像
					OutputArray sqsum, 		// 输出平方求和积分图像
					int sdepth = -1, 		// 输出标准求和积分图像的数据类型标志,-1表示满足数据存储的自适应类型
					int sqdepth = -1 		// 输出平方求和积分图像的数据类型标志,	-1表示满足数据存储的自适应类型
					);

	积分图像函数(倾斜求和积分)
	cv::integral(	InputArray src, 		// 输入图像
					OutputArray sum, 		// 输出标准求和积分图像
					OutputArray sqsum, 		// 输出平方求和积分图像
					OutputArray tilted, 	// 输出倾斜45°的倾斜求和积分图像
					int sdepth = -1, 		// 输出标准求和积分图像的数据类型标志,-1表示满足数据存储的自适应类型
					int sqdepth = -1, 		// 输出平方求和积分图像的数据类型标志,	-1表示满足数据存储的自适应类型
					);


	*/
	//创建一个512×512全为1的矩阵
	cv::Mat mat = cv::Mat::ones(512, 512, CV_32FC1);

	//在图像中加入随机噪声
	cv::RNG rng1(10000);
	for (int y = 0; y < mat.rows; y++)  
	{
		for (int x = 0; x < mat.cols; x++)
		{
			float d = rng1.uniform(-0.5, 0.5);
			mat.at<float>(y, x) = mat.at<float>(y, x) + d;
		}
	}
	
	//计算标准求和积分
	cv::Mat sum;
	cv::integral(mat, sum);
	//为了便于显示，转成CV_8U格式
	cv::Mat sum8U = cv::Mat_<uchar>(sum);

	//计算平方求和积分
	cv::Mat sqsum;
	integral(mat, sum, sqsum);
	//为了便于显示，转成CV_8U格式
	cv::Mat sqsum8U = cv::Mat_<uchar>(sqsum);

	//计算倾斜求和积分
	cv::Mat tilted;
	integral(mat, sum, sqsum, tilted);
	//为了便于显示，转成CV_8U格式
	cv::Mat tilted8U = cv::Mat_<uchar>(tilted);

	//输出结果
	cv::imshow("sum8U", sum8U);
	cv::imshow("sqsum8U", sqsum8U);
	cv::imshow("tilted8U", tilted8U);

	cv::waitKey(0);
	cv::destroyAllWindows();

	/* 图像分割 */
	/*
	漫水填充法(只能设置一个种子点)
	漫水填充函数1
	cv::floodFill(	InputOutputArray image, 		// 输入输出图像
					InputOutputArray mask, 			// 输入输出掩码图像
					Point seedPoint, 				// 种子点
					Scalar newVal, 					// 归入种子点区域内的新像素值
					Rect* rect = 0, 				// 种子点漫水填充区域的最小外接矩形
					Scalar loDiff = Scalar(), 		// 低于当前像素值的最大差值
					Scalar upDiff = Scalar(), 		// 高于当前像素值的最大差值
					int flags = 4 					// 操作标志
					);
	
	漫水填充函数2			
	cv::floodFill(	InputOutputArray image, 		// 输入输出图像
					Point seedPoint, 				// 种子点
					Scalar newVal, 					// 归入种子点区域内的新像素值
					Rect* rect = 0, 				// 种子点漫水填充区域的最小外接矩形
					Scalar loDiff = Scalar(), 		// 低于当前像素值的最大差值
					Scalar upDiff = Scalar(), 		// 高于当前像素值的最大差值
					int flags = 4 					// 操作标志
					);

	分水岭算法函数(设置多个种子点)
	cv::watershed(	InputArray image, 			// 输入图像
					InputOutputArray markers	// 输入输出标记图像
					);

	Grabcut法(使用高斯混合模型估计目标区域的前景和背景)
	cv::grabCut(	InputArray img, 				// 输入图像
					InputOutputArray mask, 			// 输入输出掩码图像
					Rect rect, 						// 包含对象的ROI矩形区域
					InputOutputArray bgdModel, 		// 输入输出背景模型
					InputOutputArray fgdModel, 		// 输入输出前景模型
					int iterCount, 					// 迭代次数
					int mode = cv::GC_EVAL 			// 操作标志
					);	
	
	Mean-Shift法
	cv::pyrMeanShiftFiltering(	InputArray src, 																		// 输入图像
								OutputArray dst, 																		// 输出图像
								double sp, 																				// 滑动窗口的半径
								double sr, 																				// 滑动窗口的颜色幅度
								int maxLevel = 1, 																		// 分割金字塔缩放层数
								TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1) 	// 终止条件
								);
	终止条件构造函数
	cv::TermCriteria::TermCriteria(	int type, 		// 终止条件的类型
									int maxCount, 	// 最大迭代次数
									double epsilon	// 精度
									);
	*/
	// 生产一个Lena图像的副本
	cv::Mat lena_copy1 = lena.clone();
	cv::RNG rng2(10000);// 随机数，用于随机生成像素

	// 设置操作标志flags
	int connectivity = 4;  // 连通邻域方式
	int maskVal = 255;  // 掩码图像的数值
	int flags = connectivity|(maskVal<<8)| cv::FLOODFILL_FIXED_RANGE;  // 漫水填充操作方式标志

	// 设置与选中像素点的差值
	cv::Scalar loDiff1 = cv::Scalar(20, 20, 20);
	cv::Scalar upDiff1 = cv::Scalar(20, 20, 20);

	// 声明掩模矩阵变量
	cv::Mat mask1 = cv::Mat::zeros(lena_copy1.rows + 2, lena_copy1.cols + 2, CV_8UC1);

	while (true)
	{
		// 随机产生图像中某一像素点
		int py = rng2.uniform(0, lena_copy1.rows-1);
		int px = rng2.uniform(0, lena_copy1.cols - 1);
		cv::Point point1 = cv::Point(px, py);

		// 彩色图像中填充的像素值
		cv::Scalar newVal = cv::Scalar(rng2.uniform(0, 255), rng2.uniform(0, 255), rng2.uniform(0, 255));

		// 漫水填充函数
		int area = cv::floodFill(lena_copy1, mask1, point1, newVal, 0, loDiff1, upDiff1, flags);

		// 输出像素点和填充的像素数目
		std::cout << "像素点x：" << point1.x << "  y:" << point1.y
			<< "     填充像数数目：" << area << std::endl;

		// 输出填充的图像结果
		cv::imshow("填充的彩色图像", lena_copy1);
		cv::imshow("掩模图像", mask1);

		// 判断是否结束程序
		int c = cv::waitKey(0);
		if ((c&255)==27)
		{
			break;
		}
	}

	cv::Mat HoughLines, HoughLinesGray, HoughLinesMask;
	cv::Mat maskWaterShed;  // watershed()函数的参数
	HoughLines = cv::imread("../Img/HoughLines.jpg");  // 原图像
	if (HoughLines.empty())
	{
		std::cout << "请确认图像文件名称是否正确" << std::endl;
		return -1;
	}
	cv::cvtColor(HoughLines, HoughLinesGray, cv::COLOR_BGR2GRAY);

	// 提取边缘并进行闭运算
	cv::Canny(HoughLinesGray, HoughLinesMask, 150, 300);

	cv::imshow("边缘图像", HoughLinesMask);
	cv::imshow("原图像", HoughLines);

	// 计算连通域数目
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(HoughLinesMask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

	// 在maskWaterShed上绘制轮廓,用于输入分水岭算法
	maskWaterShed = cv::Mat::zeros(HoughLinesMask.size(), CV_32S);
	for (int index = 0; index < contours.size(); index++)
	{
		cv::drawContours(maskWaterShed, contours, index, cv::Scalar::all(index + 1),
			-1, 8, hierarchy, INT_MAX);
	}
	// 分水岭算法   需要对原图像进行处理
	cv::watershed(HoughLines, maskWaterShed);

	std::vector<cv::Vec3b> colors;  // 随机生成几种颜色
	for (int i = 0; i < contours.size(); i++)
	{
		int b = cv::theRNG().uniform(0, 255);
		int g = cv::theRNG().uniform(0, 255);
		int r = cv::theRNG().uniform(0, 255);
		colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	cv::Mat resultImg = cv::Mat(HoughLines.size(), CV_8UC3);  // 显示图像
	for (int i = 0; i < HoughLinesMask.rows; i++)
	{
		for (int j = 0; j < HoughLinesMask.cols; j++)
		{
			// 绘制每个区域的颜色
			int index = maskWaterShed.at<int>(i, j);
			if (index == -1)  // 区域间的值被置为-1（边界）
			{
				resultImg.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
			}
			else if (index <= 0 || index > contours.size())  // 没有标记清楚的区域被置为0 
			{
				resultImg.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
			}
			else  // 其他每个区域的值保持不变：1，2，…，contours.size()
			{
				resultImg.at<cv::Vec3b>(i, j) = colors[index - 1];  // 把些区域绘制成不同颜色
			}
		}
	}

	resultImg = resultImg * 0.6 + HoughLines * 0.4;
	cv::imshow("分水岭结果", resultImg);

	// // 绘制每个区域的图像
	// for (int n = 1; n <= contours.size(); n++)
	// {
	// 	cv::Mat resImage1 = cv::Mat(HoughLines.size(), CV_8UC3);  // 声明一个最后要显示的图像
	// 	for (int i = 0; i < HoughLinesMask.rows; i++)
	// 	{
	// 		for (int j = 0; j < HoughLinesMask.cols; j++)
	// 		{
	// 			int index = maskWaterShed.at<int>(i, j);
	// 			if (index == n)
	// 				resImage1.at<cv::Vec3b>(i, j) = HoughLines.at<cv::Vec3b>(i, j);
	// 			else
	// 				resImage1.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
	// 		}
	// 	}
	// 	// 显示图像
	// 	imshow(cv::to_string(n) resImage1);
	// }
	cv::waitKey(0);
	cv::destroyAllWindows();

	// 绘制矩形
	cv::Mat lena_copy2 = lena.clone();
	cv::Mat imgRect;
	lena_copy2.copyTo(imgRect);  // 备份图像，方式绘制矩形框对结果产生影响
	cv::Rect rect(80, 30, 340, 390);
	cv::rectangle(imgRect, rect, cv::Scalar(255, 255, 255),2);
	cv::imshow("选择的矩形区域", imgRect);

	// 进行分割
	cv::Mat bgdmod = cv::Mat::zeros(1, 65, CV_64FC1);
	cv::Mat fgdmod = cv::Mat::zeros(1, 65, CV_64FC1);
	cv::Mat mask2 = cv::Mat::zeros(lena.size(), CV_8UC1);
	grabCut(lena_copy2, mask2, rect, bgdmod, fgdmod, 5, cv::GC_INIT_WITH_RECT);
	
	// 将分割出的前景绘制回来
	cv::Mat result;
	for (int row = 0; row < mask2.rows; row++) 
	{
		for (int col = 0; col < mask2.cols; col++) 
		{
			int n = mask2.at<uchar>(row, col);
			// 将明显是前景和可能是前景的区域都保留
			if (n == 1 || n == 3) 
			{
				mask2.at<uchar>(row, col) = 255;
			}
			// 将明显是背景和可能是背景的区域都删除
			else 
			{
				mask2.at<uchar>(row, col) = 0;
			}
		}
	}
	cv::bitwise_and(lena_copy2, lena_copy2, result, mask2);
	cv::imshow("分割结果", result);
	cv::waitKey(0);
	cv::destroyAllWindows();

	cv::Mat coins = cv::imread("../Img/coins.png");
	if (!coins.data)
	{
		std::cout << "读取图像错误，请确认图像文件是否正确" << std::endl;
		return -1;
	}

	// 分割处理
	cv::Mat result3, result4;
	cv::TermCriteria T10 = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.1);
	cv::pyrMeanShiftFiltering(coins, result3, 20, 40, 2, T10);  // 第一次分割
	cv::pyrMeanShiftFiltering(result3, result4, 20, 40, 2, T10);  // 第一次分割的结果再次分割

	// 显示分割结果
	cv::imshow("coins", coins);
	cv::imshow("result3", result3);
	cv::imshow("result4", result4);

	// 对图像提取Canny边缘
	cv::Mat coinsCanny, result3Canny, result4Canny;
	cv::Canny(coins, coinsCanny, 150, 300);
	cv::Canny(result3, result3Canny, 150, 300);
	cv::Canny(result4, result4Canny, 150, 300);

	// 显示边缘检测结果
	cv::imshow("coinsCanny", coinsCanny);
	cv::imshow("result3Canny", result3Canny);
	cv::imshow("result4Canny", result4Canny);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/* 图像修复 */
	/*
	图像修复函数
	cv::inpaint(InputArray src, 			// 输入图像
				InputArray inpaintMask, 	// 输入修复掩码图像
				OutputArray dst, 			// 输出图像
				double inpaintRadius, 		// 修复半径(算法考虑的每个点的圆形邻域)
				int flags 					// 修复方法
				);
	*/

	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}