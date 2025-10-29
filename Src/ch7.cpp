#include <iostream>
#include <opencv2/opencv.hpp>

void drawLine(cv::Mat &img, //要标记直线的图像
	std::vector<cv::Vec2f> lines,   //检测的直线数据
	double rows,   //原图像的行数（高）
	double cols  //原图像的列数（宽）
){
	cv::Point pt1, pt2;
	for (size_t i = 0; i < lines.size(); i++)
	{
		// 检测的直线数据为极坐标情况下
		float rho = lines[i][0];  //直线距离坐标原点的距离
		float theta = lines[i][1];  //直线过坐标原点垂线与x轴夹角
		double a = cos(theta);  //夹角的余弦值
		double b = sin(theta);  //夹角的正弦值
		double x0 = a*rho, y0 = b*rho;  //直线与过坐标原点的垂线的交点
		double length = cv::max(rows, cols);  //图像高宽的最大值
		
		//计算直线上的一点
		pt1.x = cvRound(x0 + length  * (-b));
		pt1.y = cvRound(y0 + length  * (a));
		//计算直线上另一点
		pt2.x = cvRound(x0 - length  * (-b));
		pt2.y = cvRound(y0 - length  * (a));
		
		//两点绘制一条直线
		line(img, pt1, pt2, cv::Scalar(255), 2);
	}
}

void drawLineP(cv::Mat &img, //要标记直线的图像
	std::vector<cv::Vec4i> lines   //检测的直线数据
){
	cv::Point pt1, pt2;
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::line(img, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255), 3);
	}
}

int main(){
	/* 形状检测 */
	// 霍夫变换检测是否存在直线
	// 将直线的斜率K和截距b，变换为二维坐标点（x，y）[将二维直线变换为一维坐标点]
	cv::Mat HoughLine = cv::imread("../Img/HoughLines.jpg");
	if (HoughLine.empty())
	{ 
		std::cout << "图片不存在" << std::endl;
		return -1;
	}
	// 边缘检测
	cv::Mat HoughLineEdge;
	cv::Canny(HoughLine, HoughLineEdge, 80, 180);
	cv::threshold(HoughLineEdge, HoughLineEdge, 170, 255, cv::THRESH_BINARY);

	// 霍夫变换
	/*
	标准霍夫变换和多尺度霍夫变换函数
	cv::HoughLines( InputArray image, 			// 输入图像 
					OutputArray lines,			// 霍夫变换检测到的直线极坐标描述的系数，分别表示直线距离坐标原点的距离r和坐标原点到直线的垂线与x轴的夹角theta
                	double rho, 				// 以像素为单位的距离分辨率
					double theta, 				// 以弧度为单位的角度分辨率
					int threshold,				// 累加器的阈值（有多少个坐标在同一直线上才被确认形状为直线）
                	double srn = 0, 			// 该参数表示距离分辨率的除数
					double stn = 0, 			// 该参数表示角度分辨率的除数
                	double min_theta = 0, 		// 检测直线的最小角度
					double max_theta = CV_PI 	// 检测直线的最大角度
					);
	*/
	std::vector<cv::Vec2f> line1, line2;
	cv::HoughLines(HoughLineEdge, line1, 1, CV_PI / 180, 50, 0, 0);
	cv::HoughLines(HoughLineEdge, line2, 1, CV_PI / 180, 150, 0, 0);

	// 在原图像中绘制直线
	cv::Mat img1, img2;
	img1 = HoughLine.clone();
	img2 = HoughLine.clone();
	drawLine(img1, line1, HoughLineEdge.rows, HoughLineEdge.cols);
	drawLine(img2, line2, HoughLineEdge.rows, HoughLineEdge.cols);

	// 显示图像
	cv::imshow("HoughLine", HoughLine);
	cv::imshow("edge", HoughLineEdge);
	cv::imshow("img1", img1);
	cv::imshow("img2", img2);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/*
	渐进概率式霍夫变换函数
	cv::HoughLinesP(InputArray image, 			// 输入图像
					OutputArray lines,			// 霍夫变换检测到的直线或者线段两个端点大的坐标[x_1, y_1, x_2, y_2]
                	double rho, 				// 以像素为单位的距离分辨率
					double theta, 				// 以弧度为单位的角度分辨率
					int threshold,				// 累加器的阈值（有多少个坐标在同一直线上才被确认形状为直线）
                	double minLineLength = 0, 	// 直线的最小长度
					double maxLineGap = 0 		// 同一直线上相邻的两个点之间的最大距离
					);
	*/
	std::vector<cv::Vec4i> line3, line4;
	cv::HoughLinesP(HoughLineEdge, line3, 1, CV_PI / 180, 150, 30, 10);
	cv::HoughLinesP(HoughLineEdge, line4, 1, CV_PI / 180, 150, 30, 40);

	// 在原图像中绘制直线
	cv::Mat img3, img4;
	img3 = HoughLine.clone();
	img4 = HoughLine.clone();
	drawLineP(img3, line3);
	drawLineP(img4, line4);

	// 显示图像
	cv::imshow("HoughLine", HoughLine);
	cv::imshow("edge", HoughLineEdge);
	cv::imshow("img3", img3);
	cv::imshow("img4", img4);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/*
	由坐标点找是否存在直线
	cv::HoughLinesPointSet( InputArray point, 	// 输入点的集合
							OutputArray lines, 	// 输出可能存在的直线
							int lines_max, 		// 检测直线的最大数目
							int threshold,		// 累加器的阈值（有多少个坐标在同一直线上才被确认形状为直线）
                        	double min_rho, 	// 检测直线的最小角度
							double max_rho, 	// 检测直线的最大角度
							double rho_step,	// 以像素为单位的距离分辨率
                        	double min_theta, 	// 检测直线的最小角度值
							double max_theta, 	// 检测直线的最大角度值
							double theta_step 	// 以弧度为单位的角度分辨率
							);

	最小二乘拟合直线函数
	cv::fitLine(InputArray points, 	// 输入待拟合直线的二维或者三维点集
			 OutputArray line, 	// 输出描述直线的参数
			 int distType,		// 拟合算法使用的距离类型标志
             double param, 		// 某些距离类型的数值参数
			 double reps, 		// 坐标原点与拟合直线之间的距离精度
			 double aeps 		// 拟合直线的角度精度
			 );
	
	霍夫变换检测圆形
	cv::HoughCircles( 	InputArray image, 		// 输入图像
						OutputArray circles,	// 检测结果的输出量，用三个参数表示，分别是圆心的坐标和圆的半径
                        int method, 			// 检测圆形的方法标志
						double dp, 				// 离散化时分辨率与图像分辨率的反比
						double minDist,			// 检测结果中两个圆心之间的最小距离（NMS非极大抑制化）
                        double param1 = 100, 	// 传递给Canny边缘检测器的两个阈值的较大值
						double param2 = 100,	// 检测圆形的累加器阈值
                        int minRadius = 0, 		// 检测圆的最小半径
						int maxRadius = 0 		// 检测圆的最大半径
						);
	*/

	/* 轮廓检测 */
	// 轮廓索引
	/*
	提取图像轮廓函数
	cv::findContours( 	InputArray image, 				// 输入图像
						OutputArrayOfArrays contours,	// 检测到的轮廓
                        OutputArray hierarchy, 			// 轮廓结构关系描述向量[同层下一个轮廓索引， 同层上一个轮廓索引， 下一层第一个子轮廓索引， 上层父轮廓索引]
						int mode,						// 轮廓检测模式标志
                        int method, 					// 轮廓逼近方法标志
						Point offset = Point()			// 每个轮廓点移动的可选偏移量
						);
	
	显示图像轮廓函数
	cv::drawContours( 	InputOutputArray image, 			// 输入图像
						InputArrayOfArrays contours,		// 所有将要绘制的轮廓
                        int contourIdx, 					// 要绘制的轮廓的数目
						const Scalar& color,				// 绘制轮廓的颜色
                        int thickness = 1, 					// 绘制轮廓的线条粗细
						int lineType = LINE_8,				// 边界线的连接类型
                        InputArray hierarchy = noArray(),	// 可选的结构关系信息
                        int maxLevel = INT_MAX, 			// 表示绘制轮廓的最大等级
						Point offset = Point() 				// 可选的轮廓偏移参数
						);
	*/
	cv::Mat key = cv::imread("../Img/keys.jpg");
	if (key.empty())
	{ 
		std::cout << "图片不存在" << std::endl;
		return -1;
	}
	cv::imshow("Key", key);
	cv::Mat keygray, keybinary;
	cv::cvtColor(key, keygray, cv::COLOR_BGR2GRAY);  //转化成灰度图
	cv::GaussianBlur(keygray, keygray, cv::Size(13, 13), 4, 4);  //平滑滤波
	cv::threshold(keygray, keybinary, 170, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  //自适应二值化

	std::vector<std::vector<cv::Point>> keycontours;  //轮廓
	std::vector<cv::Vec4i> keyhierarchy;  //存放轮廓结构变量
	cv::findContours(keybinary, keycontours, keyhierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());
	//绘制轮廓
	for (int t = 0; t < keycontours.size(); t++)
	{
		drawContours(key, keycontours, t, cv::Scalar(0, 0, 255), 2, 8);
	}
	//输出轮廓结构描述子
	for (int i = 0; i < keyhierarchy.size(); i++)
	{
		std::cout << keyhierarchy[i] << std::endl;
	}
	std::cout <<  std::endl;

	//显示结果
	cv::imshow("轮廓检测结果", key);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// 轮廓面积
	/*
	轮廓面积
	cv::contourArea(InputArray contour, 	// 轮廓的像素点
					bool oriented = false 	// 区域面积是否具有方向的标志，true表示具有方向性，false表示没有方向性
					);						// 返回值double数据类型，统计轮廓面积
	*/
	cv::Mat coin = cv::imread("../Img/coins.jpg");
	if (coin.empty())
	{ 
		std::cout << "图片不存在" << std::endl;
		return -1;
	}
	cv::Mat coingray, coinbinary;
	cv::cvtColor(coin, coingray, cv::COLOR_BGR2GRAY);  //转化成灰度图
	cv::GaussianBlur(coingray, coingray, cv::Size(9, 9), 2, 2);  //平滑滤波
	cv::threshold(coingray, coinbinary, 170, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  //自适应二值化

	std::vector<std::vector<cv::Point>> coincontours;  //轮廓
	std::vector<cv::Vec4i> coinhierarchy;  //存放轮廓结构变量
	cv::findContours(coingray, coincontours, coinhierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());

	//输出轮廓面积
	for (int t = 0; t < coincontours.size(); t++)
	{
		double area1 = cv::contourArea(coincontours[t]);
		std::cout << "第" << t << "轮廓面积=" << area1 << std::endl;
	}
	
	// 轮廓周长
	/*
	cv::arcLength( 	InputArray curve, 	// 轮廓或者曲线的二维像素点
					bool closed 		// 轮廓或者曲线是否闭合的标志，true表示闭合
					);					// 返回值double数据类型，统计轮廓周长
	*/
	//输出轮廓长度
	for (int t = 0; t < coincontours.size(); t++)
	{
		double length2 = cv::arcLength(coincontours[t], true);
		std::cout << "第" << t << "个轮廓长度=" << length2 << std::endl;
	}

	// 轮廓识别
	/*
	求取轮廓最大外接矩阵函数
	cv::boundingRect( 	InputArray array	// 输入灰度图像或者二维点集
						);					// 返回值Rect数据类型，[x, y , w, h]

	求取轮廓最小外接矩阵函数
	cv::minAreaRect(InputArray points 		// 输入的二维点集合
					);						// 返回值RotatedRect数据类型，输出4个顶点和中心坐标

	求取逼近轮廓的多边形函数
	cv::approxPolyDP( 	InputArray curve,			// 输入轮廓像素点
                        OutputArray approxCurve,	// 多边形逼近结果
                        double epsilon, 			// 逼近的精度，即原始曲线和逼近曲线之间的最大距离
						bool closed 				// 逼近曲线是否为封闭曲线的标志
						);
	*/

	// 点到轮廓的距离
	/*
	计算像素点距离轮廓最小距离函数
	cv::pointPolygonTest(	InputArray contour, 	// 输入轮廓
							Point2f pt, 			// 需要计算与轮廓距离的像素点
							bool measureDist 		// 计算的距离是否具有方向性的标志
							);						// 返回值double数据类型，计算像素点距离轮廓最小距离
	*/

	// 凸包（凸空间）检测
	/*
	cv::convexHull( InputArray points, 			// 输入的二维点集或轮廓坐标
					OutputArray hull,			// 输出凸包的顶点
                    bool clockwise = false, 	// 方向标志
					bool returnPoints = true 	// 输出数据的类型标志，true表示凸包顶点的坐标，false表示凸包顶点的索引
					);
	*/

	/* 矩的计算 */
	/*
	计算图像矩的函数
	cv::moments(InputArray array, 			// 计算矩的区域二维像素坐标集合 或者  单通道大的CV_8U图像
				bool binaryImage = false 	// 是否将所有非零像素值视为1的标志
				);							// 返回值Moments，含有几何矩、中心距及归一化的几何矩的数值属性
	*/

	// Hu矩具有旋转、平移和缩放不变性
	// 可以通过Hu矩实现图像轮廓的匹配
	/*
	计算图像Hu矩的函数1
	cv::HuMoments( 	const Moments& moments, // 输入的图像矩
					double hu[7] 			// 输出Hu矩的7个值
					);

	计算图像Hu矩的函数2
	cv::HuMoments( 	const Moments& m, 		// 输入的图像矩
					OutputArray hu 			// 输出Hu矩的矩阵
					);

	利用Hu矩进行轮廓匹配函数
	cv::matchShapes(InputArray contour1, 	// 原灰度图像或者轮廓
					InputArray contour2,	// 模板图像或者轮廓
                    int method, 			// 匹配方法的标志
					double parameter 		// 特定于方法的参数（现在不支持）
					);						// 返回值double数据类型，
	*/
	cv::Mat img_ABC = cv::imread("../Img/ABC.png");
	cv::Mat img_B = cv::imread("../Img/B.png");
	if (img_ABC.empty() || img_B.empty())
	{
		std::cout << "图片不存在" << std::endl;
		return -1;
	}
	cv::resize(img_B, img_B, cv::Size(), 0.5, 0.5);

	cv::Mat gray_ABC, binary_ABC, gray_B, binary_B;
	cv::cvtColor(img_ABC, gray_ABC, cv::COLOR_BGR2GRAY);  	//转化成灰度图
	cv::cvtColor(img_B, gray_B, cv::COLOR_BGR2GRAY); 	 	//转化成灰度图
	cv::GaussianBlur(gray_ABC, gray_ABC, cv::Size(13, 13), 4, 4);  	//平滑滤波
	cv::GaussianBlur(gray_B, gray_B, cv::Size(13, 13), 4, 4);  		//平滑滤波
	cv::threshold(gray_ABC, binary_ABC, 170, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  	//自适应二值化
	cv::threshold(gray_B, binary_B, 170, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  		//自适应二值化

	std::vector<std::vector<cv::Point>> contours1, contours2;  //轮廓
	std::vector<cv::Vec4i> hierarchy1, hierarchy2;  //存放轮廓结构变量
	cv::findContours(binary_ABC, contours1, hierarchy1, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());
	cv::findContours(binary_B, contours2, hierarchy2, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());

	// hu矩计算
	cv::Moments mm2 = cv::moments(contours2[0]);
	cv::Mat hu2;
	cv::HuMoments(mm2, hu2);

	// 轮廓匹配
	for (int n = 0; n < contours1.size(); n++)
	{
		cv::Moments mm = cv::moments(contours1[n]);
		cv::Mat hum;
		cv::HuMoments(mm, hum);
		//Hu矩匹配
		double dist;
		dist = cv::matchShapes(hum, hu2, cv::CONTOURS_MATCH_I1, 0);
		if (dist < 1)
		{
			cv::drawContours(img_ABC, contours1, n, cv::Scalar(0, 0, 255), 3, 8);
		}
	}
	cv::imshow("match result", img_ABC);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/* 点集拟合 */
	/*
	寻找二维点集的最小包围三角形
	cv::minEnclosingTriangle( 	InputArray points, 			// 待寻找包围三角形的二位点集
								CV_OUT OutputArray triangle // 拟合出的三角形的3个顶点坐标
								);							// 返回值double数据类型，三角形面积
	
	寻找二维点集的最小包围圆形
	cv::minEnclosingCircle( InputArray points,				// 待寻找包围圆形的二位点集
                            CV_OUT Point2f& center, 		// 圆形的圆心
							CV_OUT float& radius 			// 圆形的半径
							);

	*/

	/* QR二维码检测 */
	/*
	定位QR二维码函数
	cv::QRCodeDetector::detect(	InputArray img, 								// 待检测是否含有QR二维码的灰度图像或者彩色图像
								OutputArray points								// 包含QR二维码的最小区域四边形的4个顶点坐标
								) const;										// 返回值bool数据类型，表示是否含有二维码的结果，true表示有，false表示无

	解码QR二维码函数
	cv::QRCodeDetector::decode(	InputArray img, 								// 含有QR二维码的图像
								InputArray points, 								// 包含QR二维码的最小区域的四边形
								OutputArray straight_code = noArray()			// 经过校正和二值化的QR二维码（变换后的QR二维码）
								) const;										// 返回值string数据类型，输出解码结果

	同时定位和解码QR二维码函数
	cv::QRCodeDetector::detectAndDecode(InputArray img, 						// 含有QR二维码的图像
										OutputArray points = noArray(),			// 包含QR二维码的最小区域的四边形的4个顶点坐标
                                        OutputArray straight_code = noArray()	// 经过校正和二值化的QR二维码（变换后的QR二维码）
										) const;								// 返回值string数据类型，输出解码结果
	*/
	cv::Mat qrcode = cv::imread("../Img/qrcode2.png");
	if (qrcode.empty())
	{
		std::cout << "图片不存在" << std::endl;
		return -1;
	}
	cv::Mat qrcodegray, qrcode_bin;
	cv::cvtColor(qrcode, qrcodegray, cv::COLOR_BGR2GRAY);
	cv::QRCodeDetector qrcodedetector;
	std::vector<cv::Point> points;
	std::string information;
	bool isQRcode;
	isQRcode = qrcodedetector.detect(qrcodegray, points);
	if(isQRcode){
		// 解码二维码
		information = qrcodedetector.decode(qrcodegray, points, qrcode_bin);
		std::cout << points << std::endl;
		std::cout << std::endl;
	}
	else{
		std::cout << "无法识别二维码" << std::endl;
		return -1;
	}
	
	//绘制二维码的边框
	for (int i = 0; i < points.size(); i++)
	{
		if (i == points.size() - 1)
		{
			cv::line(qrcode, points[i], points[0], cv::Scalar(0, 0, 255), 2, 8);
			break;
		}
		cv::line(qrcode, points[i], points[i + 1], cv::Scalar(0, 0, 255), 2, 8);
	}
	//将解码内容输出到图片上
	cv::putText(qrcode, information.c_str(), cv::Point(20, 30), 0, 1.0, cv::Scalar(0, 0, 255), 2, 8);

	//输出结果
	cv::imshow("result", qrcode);
	cv::imshow("qrcode_bin", qrcode_bin);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}