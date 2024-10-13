#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
	// 读取图片，图片格式为RBG颜色模型
	cv::Mat RGB = cv::imread("../Img/Lena.jpg", cv::IMREAD_COLOR);
	// 判断图片是否存在
	if (RGB.empty())
	{
		std::cout << "图片不存在！" << std::endl;
		return -1;
	}

	/* 图像颜色空间 */
	// 转换图片颜色格式
	cv::Mat gray, HSV, YUV, Lab, RGB32;
	/*
	图片变换
	Mat.convertTo(	OutputArray dst,	转换类型后输出的图像
					int rtype, 			转换图像的数据类型
					double alpha, 		转换过程中的缩放因子
					double beta			转换过程中的偏置因子
					);
	*/
	RGB.convertTo(RGB32, CV_32F, 1.0 / 255); // 将CV_8U类型转换为CV_32F类型(将数值范围由[0,255]映射到[0,1])
	// img32.convertTo(RGB, CV_8U, 255);			// 将CV_32F类型转换为CV_8U类型(将数值范围由[0,1]映射到[0,255])

	/*
	转换图片颜色格式
	cvtColor( 	InputArray src, 	待转换颜色模型的原始图像
				OutputArray dst, 	转换颜色模型后的目标图像
				int code, 			颜色空间转换的标志
				int dstCn = 0 		目标图像中的通道数
				);
	*/
	cv::cvtColor(RGB32, gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(RGB32, HSV, cv::COLOR_BGR2HSV);
	cv::cvtColor(RGB32, YUV, cv::COLOR_BGR2YUV);
	cv::cvtColor(RGB32, Lab, cv::COLOR_BGR2Lab);

	// 展示图片
	cv::imshow("原图", RGB32);
	cv::imshow("灰色图", gray);
	cv::imshow("HSV图", HSV);
	cv::imshow("YUV图", YUV);
	cv::imshow("Lab图", Lab);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// 多通道分离与合并
	cv::Mat imgs[3];
	std::vector<cv::Mat> imgv;
	/*
	多通道分离
	split(	const Mat& src, 		带分离的多通道图像
			Mat* mvbegin			分离后的单通道图像，为数组格式
			);
	split(	InputArray m, 			带分离的多通道图像
			OutputArrayOfArrays mv	分离后的单通道图像，为向量(vector)形式
			);
	*/
	cv::split(RGB, imgs);
	cv::split(HSV, imgv);

	cv::Mat com_imgs, com_imgv;
	/*
	多通道合并
	merge(	const Mat* mv, 			需要合并的图像数组
			size_t count, 			输入的图像数组的长度，其数值必须大于0
			OutputArray dst			合并后的图像
			);
	merge(	InputArrayOfArrays mv, 	需要合并的图像向量
			OutputArray dst			合并后的图像
			);
	*/
	cv::merge(imgs, 3, com_imgs);
	cv::merge(imgv, com_imgv);

	cv::imshow("RGB-B通道", imgs[0]);
	cv::imshow("RGB-G通道", imgs[1]);
	cv::imshow("RGB-R通道", imgs[2]);

	cv::imshow("HSV-H通道", imgv[0]);
	cv::imshow("HSV-S通道", imgv[1]);
	cv::imshow("HSV-V通道", imgv[2]);

	cv::imshow("RGB合并图", com_imgs);
	cv::imshow("HSV合并图", com_imgv);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/* 像素操作处理 */
	/*
	寻找图像像素的最大值与最小值
	*/
	double minVal, maxVal;
	cv::Point minIdx, maxIdx;
	/*
	计算矩阵中的最大值与最小值
	minMaxLoc(	InputArray src, 				目标矩阵，必须是单通道矩阵
				CV_OUT double* minVal,			矩阵中的最小值
				CV_OUT double* maxVal = 0, 		矩阵中的最大值
				CV_OUT Point* minLoc = 0,		矩阵中的最小值在矩阵中的坐标
				CV_OUT Point* maxLoc = 0, 		矩阵中的最大值在矩阵中的坐标
				InputArray mask = noArray()		掩膜(设置矩阵中的指定区域)
				);

	拓展矩阵(将多通道矩阵变为单通道矩阵)
	Mat.reshape(int cn, 						转换后矩阵的通道数
				int rows=0						转换后矩阵的行数
				);
	*/
	cv::minMaxLoc(gray, &minVal, &maxVal, &minIdx, &maxIdx);

	// 计算图像的平均值和标准差
	// 标准差表示图像中明暗变化的对比程度，标准差越大，表示图像中敏感变化越明显
	/*
	计算矩阵的平均值
	mean(	InputArray src, 				目标矩阵
			InputArray mask = noArray()		掩膜
			);								返回值cv::Scalar，得到平均值

	计算矩阵的标准差
	meanStdDev(	InputArray src, 			目标矩阵
				OutputArray mean, 			图片每个通道的平均值
				OutputArray stddev,			图片每个通道的标准差
				InputArray mask=noArray()	掩膜
				);

	最大矩阵
	max(const Mat& src1, 		第一个图像矩阵
		const Mat& src2, 		第二个图像矩阵
		Mat& dst				保留对应位置较大灰度值的图像矩阵
		);

	最小矩阵
	min(const Mat& src1, 		第一个图像矩阵
		const Mat& src2, 		第二个图像矩阵
		Mat& dst				保留对应位置较小灰度值的图像矩阵
		);

	逻辑运算"与"
	bitwise_and(InputArray src1, 				第一个图像矩阵
				InputArray src2,				第二个图像矩阵
				OutputArray dst, 				逻辑运算输出结果
				InputArray mask = noArray()		掩膜
				);

	逻辑运算"或"
	bitwise_or(	InputArray src1, 				第一个图像矩阵
				InputArray src2,				第二个图像矩阵
				OutputArray dst,				逻辑运算输出结果
				InputArray mask = noArray()		掩膜
				);

	逻辑运算"异或"
	bitwise_xor(InputArray src1, 				第一个图像矩阵
				InputArray src2,				第二个图像矩阵
				OutputArray dst,				逻辑运算输出结果
				InputArray mask = noArray()		掩膜
				);

	逻辑运算"非"
	bitwise_not(InputArray src, 				第一个图像矩阵
				OutputArray dst,				逻辑运算输出结果
				InputArray mask = noArray()		掩膜
				);

	*/

	// 图像二值化(使用"阈值"和"查找表")
	/*
	二值化方法
	threshold( 	InputArray src, 	输入图像
				OutputArray dst,	二值化后的图像
				double thresh, 		二值化阈值
				double maxval, 		二值化过程中的最大值
				int type 			选择图像二值化方法的标志
				);					返回值double,根据图像二值化方法得到相关坐标点的元素

	局部自适应阈值的二值化方法
	adaptiveThreshold( 	InputArray src, 	输入图像
						OutputArray dst,	二值化后的图像
						double maxValue, 	二值化的最大值
						int adaptiveMethod,	自适应确定阈值的方法
						int thresholdType, 	选择图像二值化方法的标志
						int blockSize, 		自适应确定阈值的像素领域大小
						double C 			从平均值或者加权平均值中减去的常数，可正可负
						);

	查找表
	LUT(InputArray src, 输入图像
		InputArray lut, 灰度值查找表
		OutputArray dst	输出图像
		);
	*/

	/* 图像变换 */
	// 图像连接
	/*
	上下连接1
	vconcat(const Mat* src, 	输入图像(确定拼接图像的宽度)
			size_t nsrc, 		对输入图像进行n次上下拼接
			OutputArray dst		输出图像
			);

	上下连接2
	vconcat(InputArray src1,	输入图像1(确定拼接图像的宽度)
			InputArray src2,	输入图像2
			OutputArray dst		输出图像
			);

	左右连接1
	hconcat(const Mat* src, 	输入图像(确定拼接图像的高度)
			size_t nsrc, 		对输入图像进行n次左右拼接
			OutputArray dst		输出图像
			);

	左右连接2
	hconcat(InputArray src1, 	输入图像1(确定拼接图像的高度)
			InputArray src2, 	输入图像2
			OutputArray dst		输出图像
			);

	图像尺寸变换(先"缩放"再"调整")
	resize( InputArray src, 					输入图像
			OutputArray dst,					输出图像(要与输入图像是相同的数据类型)
			Size dsize, 						输出图像的尺寸
			double fx = 0, 						水平轴的比例因子
			double fy = 0,						垂直轴的比例因子
			int interpolation = INTER_LINEAR 	插值方法的标志
			);


	翻转变换
	flip(	InputArray src, 	输入图像
			OutputArray dst, 	输出图像
			int flipCode		翻转方式得标志(数值大于0表示绕y轴翻转;数等于0表示绕x轴翻转;数值小于0表示绕xy轴翻转)
			);

	计算旋转矩阵
	getRotationMatrix2D(Point2f center, 矩阵的旋转中心位置
						double angle, 	矩阵的旋转角度
						double scale	两个轴的比例因子
						);				返回值Mat,一个2x3矩阵

	仿射变换(综合了平移、旋转、缩放的功能)
	warpAffine( InputArray src, 						输入图像
				OutputArray dst,						输出图像
				InputArray M, 							仿射变换矩阵(综合了平移、旋转、缩放)
				Size dsize,								输出图像的尺寸
				int flags = INTER_LINEAR,				插值方法的标志
				int borderMode = BORDER_CONSTANT,		像素边界外推方法的标志
				const Scalar& borderValue = Scalar());	填充边界使用的数值，默认情况下为0

	反求仿射变换矩阵
	getAffineTransform( const Point2f src[], 	输入图像的像素坐标轴
						const Point2f dst[] 	输出图像的像素坐标轴
						);						返回值Mat,一个2x3矩阵

	透视变换
	warpPerspective(InputArray src, 						输入图像
					OutputArray dst,						输出图像
					InputArray M, 							透视变换矩阵
					Size dsize,								输出图像的尺寸
					int flags = INTER_LINEAR,				插值方法的标志
					int borderMode = BORDER_CONSTANT,		像素边界外推方法的标志
					const Scalar& borderValue = Scalar());	填充边界使用的数值，默认情况下为0

	反求透视变换矩阵
	getPerspectiveTransform(const Point2f src[], 			输入图像的像素坐标轴
							const Point2f dst[], 			输出图像的像素坐标轴
							int solveMethod = DECOMP_LU		选择计算透视变换矩阵方法的标志
							);								返回值Mat,一个3x3矩阵

	极坐标变换
	warpPolar(	InputArray src, 	输入图像
				OutputArray dst, 	输出图像
				Size dsize,			输出图像的尺寸
				Point2f center, 	极坐标变换时极坐标的原点坐标
				double maxRadius, 	变换时边界圆的半径
				int flags			插值方法与极坐标映射方法的标志
				);
	*/

	/* 在图像上绘制几何图形 */
	/*
	绘制圆形
	circle(	InputOutputArray img, 	输入图像
			Point center, 			圆形的圆心位置坐标
			int radius,				圆形的半径
			const Scalar& color, 	颜色
			int thickness = 1,		轮廓的宽度(数值为负,则绘制实心)
			int lineType = LINE_8, 	边界类型
			int shift = 0			限制数据的小数位数
			);

	绘制直线
	line(	InputOutputArray img, 	输入图像
			Point pt1, 				直线的起始点
			Point pt2, 				直线的终止点
			const Scalar& color,	颜色
			int thickness = 1, 		轮廓的宽度
			int lineType = LINE_8, 	边界类型
			int shift = 0			限制数据的小数位数
			);

	绘制椭圆
	ellipse(InputOutputArray img, 	输入图像
			Point center, 			椭圆的中心坐标
			Size axes,				椭圆的长短轴(Size(_x, _y))
			double angle, 			椭圆的旋转角度
			double startAngle, 		椭圆弧的起始角度
			double endAngle,		椭圆弧的终止角度
			const Scalar& color, 	颜色
			int thickness = 1,		轮廓的宽度(数值为负,则绘制实心)
			int lineType = LINE_8, 	边界类型
			int shift = 0			限制数据的小数位数
			);

	绘制矩形
	rectangle(	InputOutputArray img, 	输入图像
				Point pt1, 				矩阵的一个顶点
				Point pt2,				矩阵的对角线顶点(两个坐标点,不需要满足左上角、右下角原则)
				const Scalar& color, 	颜色
				int thickness = 1,		轮廓的宽度(数值为负,则绘制实心)
				int lineType = LINE_8, 	边界类型
				int shift = 0			限制数据的小数位数
				);

	绘制多边形
	fillPoly(	InputOutputArray img, 	输入图像
				const Point** pts,		多边形的顶点数组(请按照顺时针或者逆时针存放)
				const int* npts, 		每个多边形顶点数组中顶点的个数
				int ncontours,			绘制多边形的个数
				const Scalar& color, 	颜色
				int lineType = LINE_8, 	边界类型
				int shift = 0,			限制数据的小数位数
				Point offset = Point() 	所有顶点的可选偏移
				);

	文字生成
	putText( 	InputOutputArray img, 			输入图像
				const String& text, 			文本
				Point org,						图像中文字字符串的 左下角 像素坐标
				int fontFace, 					字体类型的选择标志
				double fontScale, 				字体的大小
				Scalar color,					颜色
				int thickness = 1, 				轮廓的宽度
				int lineType = LINE_8,			边界类型
				bool bottomLeftOrigin = false 	图像数据原点的位置,默认为左上角;如果参数改为true,则原点为左下角
				);
	*/
	cv::Mat Drawing = cv::Mat::zeros(cv::Size(512, 512), CV_8UC3);
	// 绘制圆形
	cv::circle(Drawing, cv::Point(50, 50), 25, cv::Scalar(255, 0, 0), -1);
	cv::circle(Drawing, cv::Point(512 - 50, 50), 25, cv::Scalar(255, 0, 0), 4);
	// 绘制直线
	cv::line(Drawing, cv::Point(0, 0), cv::Point(512, 512), cv::Scalar(255, 255, 255), 2);
	cv::line(Drawing, cv::Point(512, 0), cv::Point(0, 512), cv::Scalar(255, 255, 255), 2);
	// 绘制椭圆
	cv::ellipse(Drawing, cv::Point(256, 300), cv::Size(100, 50), 0, 0, 360, cv::Scalar(0, 255, 128), -1);
	// 绘制矩形
	cv::rectangle(Drawing, cv::Point(100, 512), cv::Point(412, 400), cv::Scalar(0, 0, 255), -1);
	cv::rectangle(Drawing, cv::Rect(200, 200, 100, 100), cv::Scalar(0, 255, 255), 3);
	// 绘制多边形
	cv::Point pp[2][5];
	pp[0][0] = cv::Point(100, 100);
	pp[0][1] = cv::Point(200, 100);
	pp[0][2] = cv::Point(200, 200);
	pp[0][3] = cv::Point(100, 200);

	pp[1][0] = cv::Point(412, 100);
	pp[1][1] = cv::Point(312, 100);
	pp[1][2] = cv::Point(312, 200);
	pp[1][3] = cv::Point(412, 200);

	cv::Point pp2[5];
	pp2[0] = cv::Point(100, 300);
	pp2[1] = cv::Point(412, 300);
	pp2[2] = cv::Point(412, 400);
	pp2[3] = cv::Point(256, 450);
	pp2[4] = cv::Point(100, 400);

	const cv::Point *pts[3] = {pp[0], pp[1], pp2};
	int npts[3] = {4, 4, 5};
	cv::fillPoly(Drawing, pts, npts, 3, cv::Scalar(255, 0, 255));

	cv::imshow("Drawing", Drawing);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/* 感兴趣区域 */
	// 从原图中截取部分内容，以减少需要处理的图像矩阵，减轻程序内存的负载
	/*
	矩阵变量
	cv::Rect(	_Tp _x, 	矩阵的左上角像素x坐标
				_Tp _y, 	矩阵的左上角像素y坐标
				_Tp _width, 矩阵的宽
				_Tp _height	矩阵的高
				)

	区间变量
	cv::Range(	int start,	区间的起始
				int end		区间的结束
				);

	例如: img(cv::Rect(200, 200, 100, 100)) 与 img(cv::Range(200, 300), cv::Range(200, 300)) 等同
	*/
	/* 图像金字塔 */
	/*
	下采样
	pyrDown(InputArray src, 					输入图像
			OutputArray dst,					输出图像
			const Size& dstsize = Size(), 		输出图像的尺寸
			int borderType = BORDER_DEFAULT 	像素边界外推方法的标志
			);

	上采样
	pyrUp( 	InputArray src, 					输入图像
			OutputArray dst,					输出图像
			const Size& dstsize = Size(), 		输出图像的尺寸
			int borderType = BORDER_DEFAULT 	像素边界外推方法的标志
			);

	*/
	std::vector<cv::Mat> Gauss, Lap; // 高斯"金字塔" 和 拉普拉斯"金字塔"
	// 高斯金字塔(解决"尺度不确定性")
	int level = 3; // 高斯"金字塔"下采样次数
	Gauss.push_back(RGB);
	// 构建高斯"金字塔"
	for (int i = 0; i < level; i++)
	{
		cv::Mat gauss;
		cv::pyrDown(Gauss[i], gauss); // 下采样
		Gauss.push_back(gauss);
	}

	// 拉普拉斯金字塔
	// 构建拉普拉斯"金字塔"
	for (int i = Gauss.size() - 1; i >= 0; i--)
	{
		cv::Mat lap, upGauss;
		if (i == Gauss.size() - 1)
		{ // 判断是否为高斯"金字塔"的最上面一层图像
			cv::Mat down;
			cv::pyrDown(Gauss[i], down); // 下采样
			cv::pyrUp(down, upGauss);	 // 上采样
			lap = Gauss[i] - upGauss;
			Lap.push_back(lap);
		}
		else
		{
			cv::pyrUp(Gauss[i + 1], upGauss);
			lap = Gauss[i] - upGauss;
			Lap.push_back(lap);
		}
	}

	// 查看两个图像"金字塔"中的图像
	for (int i = 0; i < Gauss.size(); i++)
	{
		std::string name = std::to_string(i);
		cv::imshow("G" + name, Gauss[i]);
		cv::imshow("L" + name, Lap[i]);
	}

	cv::waitKey(0);
	cv::destroyAllWindows();

	/* 窗口交互操作 */
	/*
	int createTrackbar(	const String& trackbarname, 	滑动条的名称
						const String& winname,			创建滑动条窗口的名称
                        int* value, 					指向整数变量的指针
						int count,						滑动条的最大取值
                        TrackbarCallback onChange = 0,	滑动条的最大取值
                        void* userdata = 0				传递给回调函数的可选参数
						);
	
	创建鼠标回调函数
	void setMouseCallback(	const String& winname, 		添加鼠标响应的窗口的名字
							MouseCallback onMouse, 		鼠标响应的回调函数
							void* userdata = 0			传递给回调函数的可选参数
							);

	鼠标回调函数
	void (*MouseCallback)(	int event, 					鼠标响应事件标志
							int x, 						鼠标指针在图像坐标系中的x坐标
							int y, 						鼠标指针在图像坐标系中的y坐标
							nt flags, 					鼠标响应标志
							void* userdata				传递给回调函数的可选参数
							);
	*/

	return 0;
}