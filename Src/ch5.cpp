#include <iostream>
#include <opencv2/opencv.hpp>

// 椒盐噪声产生函数
void saltAndPepper(cv::Mat img, int n)	// 传入图像以及噪声个数
{
	for (int k = 0; k<n / 2; k++)
	{
		// 随机确定图像中位置
		int i, j;
		i = std::rand() % img.cols;  //取余数运算，保证在图像的列数内 
		j = std::rand() % img.rows;  //取余数运算，保证在图像的行数内 
		int write_black = std::rand() % 2;  //判定为白色噪声还是黑色噪声的变量
		if (write_black == 0)  //添加白色噪声
		{
			if (img.type() == CV_8UC1)  //处理灰度图像
			{
				img.at<uchar>(j, i) = 255;  //白色噪声
			}
			else if (img.type() == CV_8UC3)  //处理彩色图像
			{
				img.at<cv::Vec3b>(j, i)[0] = 255; //cv::Vec3b为opencv定义的一个3个值的向量类型  
				img.at<cv::Vec3b>(j, i)[1] = 255; //[]指定通道，B:0，G:1，R:2  
				img.at<cv::Vec3b>(j, i)[2] = 255;
			}
		}
		else  //添加黑色噪声
		{
			if (img.type() == CV_8UC1)
			{
				img.at<uchar>(j, i) = 0;
			}
			else if (img.type() == CV_8UC3)
			{
				img.at<cv::Vec3b>(j, i)[0] = 0; //cv::Vec3b为opencv定义的一个3个值的向量类型  
				img.at<cv::Vec3b>(j, i)[1] = 0; //[]指定通道，B:0，G:1，R:2  
				img.at<cv::Vec3b>(j, i)[2] = 0;
			}
		}

	}
}

int main()
{
	/* 图像卷积 */
	/*
	卷积运算
	cv::filter2D(	InputArray src, 					输入图像
					OutputArray dst, 					输出图像，应与输入图像有相同的尺寸和通道数
					int ddepth,							输出图像的数据结构，当赋值为-1时，输出图像的数据类型自动选择
					InputArray kernel, 					卷积核（卷积核若不是中心对称，需要中心翻转180°后赋值于该参数）
					Point anchor = Point(-1,-1),		卷积核的锚点。锚点是卷积核中与进行处理的像素重合的点，默认值为（-1，-1）表示锚点是卷积核中心点
					double delta = 0, 					偏置
					int borderType = BORDER_DEFAULT		像素外推法选择标志
					);
	*/

	// 待卷积矩阵
	cv::Mat mat1 = (cv::Mat_<float>(5, 5) << 
					1, 2, 3, 4, 5,
				   	6, 7, 8, 9, 10,
				   11, 12, 13, 14, 15,
				   16, 17, 18, 19, 20,
				   21, 22, 23, 24, 25);

	// 卷积核（通常为3x3,5x5）
	cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
					  1, 2, 1,
					  2, 0, 2,
					  1, 2, 1);

	cv::Mat kernel_norm = kernel / 12;

	cv::Mat result, result_norm;
	cv::filter2D(mat1, result, CV_32F, kernel, cv::Point(-1, -1), 2, cv::BORDER_CONSTANT);
	cv::filter2D(mat1, result_norm, CV_32F, kernel_norm, cv::Point(-1, -1), 2, cv::BORDER_CONSTANT);
	std::cout << "result: " << std::endl
			  << result << std::endl;
	std::cout << "result_norm: " << std::endl
			  << result_norm << std::endl;
	std::cout << std::endl;

	/* 图像卷积 */
	cv::Mat Lena = cv::imread("../Img/Lena.jpg");
	cv::Mat Lena_gray;
	cv::cvtColor(Lena, Lena_gray, cv::COLOR_BGR2GRAY);

	if(Lena.empty()){
		std::cout << "图像不存在" << std::endl;
		return -1;
	}
	cv::Mat Lena_filter;
	cv::filter2D(Lena, Lena_filter, -1, kernel_norm, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	// 卷积后图像变得模糊
	cv::imshow("Lena1", Lena);
	cv::imshow("Lena_filter", Lena_filter);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/* 噪声 */
	// 椒盐噪声（脉冲噪声、随机噪声）
	/*
	三种随机数生成函数
	ing cvflann::rand();	无输入，直接范围int类型随机数

	double cvflann::rand_double(double high = 1.0,	随机数上界
								double low = 0 		随机数下界
								);					返回值double类型随机数

	int cvflann::rand_int(	int high = RAND_MAX,	随机数上界
							int low = 0 			随机数下界
							);						返回值int类型随机数
	*/
	// Lena_salt = Lena 的方式只是赋值了Mat类的矩阵头，矩阵指针指向的是同一个地址。
	// 如果希望复制两个一摸一样的Mat类而彼此之间不会受影响，那么可以使用 m = a.clone() 实现
	cv::Mat Lena_salt = Lena.clone();				// 椒盐噪声下的彩色图
	cv::Mat Lena_gray_salt = Lena_gray.clone();		// 椒盐噪声下的灰度图
	cv::imshow("Lena2", Lena_salt);
	cv::imshow("Lena_gray2", Lena_gray_salt);
	saltAndPepper(Lena_salt, 10000);		// 加入椒盐噪声
	saltAndPepper(Lena_gray_salt, 10000);	// 加入椒盐噪声
	// 显示添加椒盐噪声的图像
	cv::imshow("Lena_salt", Lena_salt);
	cv::imshow("Lena_gray_slat", Lena_gray_salt);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// 高斯噪声（噪声服从高斯分布/正态分布）
	// 高斯分布=正态分布
	/*
	cv::RNG::fill( 	InputOutputArray mat, 			存放随机数的矩阵
					int distType, 					随机数分布形式选择标志
					InputArray a, 					确定分布规律的参数
					InputArray b, 					确定分布规律的参数
					bool saturateRange = false 		预饱和标志，仅用于均匀分布
					);
	*/
	cv::Mat Lena_gauss = Lena.clone();					// 高斯噪声下的彩色图
	cv::Mat Lena_gray_gauss = Lena_gray.clone();		// 高斯噪声下的灰度图
	cv::Mat Lena_noise = cv::Mat::zeros(Lena.rows, Lena.cols, Lena.type());							// 存放高斯噪声
	cv::Mat Lena_gray_noise = cv::Mat::zeros(Lena_gray.rows, Lena_gray.cols, Lena_gray.type());		// 存放高斯噪声
	cv::imshow("Lena3", Lena_gauss);
	cv::imshow("Lena_gray3", Lena_gray_gauss);
	// 由于RNG类是非静态成员函数，因此使用时，需要创建一个RNG类的变量，通过访问这个变量的函数方式采用调用fill函数
	cv::RNG noise; // 创建一个RNG类，用于存放高斯噪声
	noise.fill(Lena_noise, cv::RNG::NORMAL, 10, 20);			// 生成三通道的高斯分布随机数
	noise.fill(Lena_gray_noise, cv::RNG::NORMAL, 15, 30);		// 生成单通道的高斯分布随机数
	cv::imshow("Lena_noise", Lena_noise);
	cv::imshow("Lena_gray_noise", Lena_gray_noise);
	Lena_gauss += Lena_noise;
	Lena_gray_gauss += Lena_gray_noise;
	// 显示添加高斯噪声后的图像
	cv::imshow("Lena_gauss", Lena_gauss);
	cv::imshow("Lena_gray_gauss", Lena_gray_gauss);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/* 线性滤波 */
	// 图像滤波和图像的卷积操作相似，但不需要将滤波模板中心旋转
	// 均值滤波（通过卷积核求得 平均值 作为滤波后的结果）
	/*
	均值滤波
	cv::blur(	InputArray src, 					输入图像
				OutputArray dst,					输出图像
                Size ksize, 						卷积核尺寸（滤波器尺寸）
				Point anchor = Point(-1,-1),		卷积核的锚点
                int borderType = BORDER_DEFAULT 	像素外推法选择标志
				);
	*/
	cv::Mat Lena_2 = Lena.clone();				// 原图
	cv::Mat Lena_salt_noise = Lena_salt.clone();	// 加入椒盐噪声
	cv::Mat Lena_gauss_noise = Lena_gauss.clone();	// 加入高斯噪声

	cv::Mat blur_3, blur_9;						// 存放原图滤波后结果，后面的数字代表滤波器尺寸
	cv::Mat blur_3salt, blur_9salt;				// 存放加入椒盐噪声的图像滤波后结果，后面的数字代表滤波器尺寸
	cv::Mat blur_3gauss, blur_9gauss;			// 存放加入椒盐噪声的图像滤波后结果，后面的数字代表滤波器尺寸
	// 调用均值滤波函数blur()进行滤波
	cv::blur(Lena_2, blur_3, cv::Size(3, 3));
	cv::blur(Lena_2, blur_9, cv::Size(9, 9));
	cv::blur(Lena_salt_noise, blur_3salt, cv::Size(3, 3));
	cv::blur(Lena_salt_noise, blur_9salt, cv::Size(9, 9));
	cv::blur(Lena_gauss_noise, blur_3gauss, cv::Size(3, 3));
	cv::blur(Lena_gauss_noise, blur_9gauss, cv::Size(9, 9));
	// 显示不含噪声的图像
	cv::imshow("Lena4", Lena_2);
	cv::imshow("Lena_3x3", blur_3);
	cv::imshow("Lena_9x9", blur_9);
	// 显示含有椒盐噪声的图像
	cv::imshow("Lena_salt_noise", Lena_salt_noise);
	cv::imshow("Lena_salt_3x3", blur_3salt);
	cv::imshow("Lena_salt_9x9", blur_9salt);
	// 显示含有高斯噪声的图像
	cv::imshow("Lena_gauss_noise", Lena_gauss_noise);
	cv::imshow("Lena_gauss_3x3", blur_3gauss);
	cv::imshow("Lena_gauss_9x9", blur_9gauss);
	cv::waitKey(0);
	cv::destroyAllWindows();
	// 方框滤波
	/*
	方框滤波（通过卷积核求得 总和 作为滤波后的结果，如果进行归一化处理则就是均值滤波）
	cv::boxFilter( 	InputArray src, 					输入图像
					OutputArray dst, 					输出图像
					int ddepth,							输出图像的数据类型
                    Size ksize, 						卷积核尺寸（滤波器尺寸）
					Point anchor = Point(-1,-1),		卷积核的锚点
                    bool normalize = true,				是否将卷积核进行归一化标志，默认参数为true，表示进行归一化
                    int borderType = BORDER_DEFAULT 	像素外推法选择标志
					);
	
	方框滤波（通过卷积核求得 平方之和 作为滤波后的结果）
	cv::boxFilter( 	InputArray src, 					输入图像
					OutputArray dst, 					输出图像
					int ddepth,							输出图像的数据类型
                    Size ksize, 						卷积核尺寸（滤波器尺寸）
					Point anchor = Point(-1,-1),		卷积核的锚点
                    bool normalize = true,				是否将卷积核进行归一化标志，默认参数为true，表示进行归一化
                    int borderType = BORDER_DEFAULT 	像素外推法选择标志
					);
	*/
	// 方框滤波不进行归一化，容易超过边界大于255，呈现白色
	
	// 待卷积矩阵（再次使用该矩阵）
	cv::Mat mat2 = (cv::Mat_<float>(5, 5) << 
					1, 2, 3, 4, 5,
				   	6, 7, 8, 9, 10,
				   11, 12, 13, 14, 15,
				   16, 17, 18, 19, 20,
				   21, 22, 23, 24, 25);

	cv::Mat Lena_32F;
	Lena.convertTo(Lena_32F, CV_32F, 1.0/255);
	// 存放方框滤波结果，方框滤波归一化结果，Sqr方框滤波结果，Sqr方框滤波归一化结果，SqrLena图
	cv::Mat box, boxNorm, Sqrbox, SqrboxNorm, SqrLena, SqrLenaNorm;
	cv::boxFilter(Lena_32F, box, -1, cv::Size(3, 3), cv::Point(-1, -1), false, cv::BORDER_CONSTANT);
	cv::boxFilter(Lena_32F, boxNorm, -1, cv::Size(3, 3), cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
	cv::sqrBoxFilter(mat2, Sqrbox, -1, cv::Size(3, 3), cv::Point(-1, -1), false, cv::BORDER_CONSTANT);
	cv::sqrBoxFilter(mat2, SqrboxNorm, -1, cv::Size(3, 3), cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
	cv::sqrBoxFilter(Lena_32F, SqrLena, -1, cv::Size(3, 3), cv::Point(-1, -1), false, cv::BORDER_CONSTANT);
	cv::sqrBoxFilter(Lena_32F, SqrLenaNorm, -1, cv::Size(3, 3), cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
	
	std::cout << "Sqrbox: " << std::endl
			  << Sqrbox << std::endl;
	std::cout << "SqrboxNorm: " << std::endl
			  << SqrboxNorm << std::endl;
	std::cout << std::endl;

	// 显示处理结果
	cv::imshow("Lena5", Lena_32F);
	cv::imshow("boxFilter", box);
	cv::imshow("boxNorm", boxNorm);
	cv::imshow("SqrLena", SqrLena);
	cv::imshow("SqrLenaNorm", SqrLenaNorm);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// 高斯滤波（高斯分布/正态分布的均值）
	/*
	高斯滤波
	cv::GaussianBlur( 	InputArray src, 					输入图像
						OutputArray dst, 					输出图像
						Size ksize,							高斯滤波器的尺寸
                        double sigmaX, 						X方向的高斯滤波器标准偏差
						double sigmaY = 0,					Y方向的高斯滤波器标准偏差
                        int borderType = BORDER_DEFAULT 	像素外推法选择标志
						);
	
	生成单一方向高斯滤波器（Y方向，滤波器尺寸为Kx1）
	cv::getGaussianKernel(	int ksize,				高斯滤波器的尺寸
							double sigma,			高斯滤波的标准差
							int ktype = CV_64F		滤波器系数的数据类型
							);
	*/
	cv::Mat Lena_3 = Lena.clone();
	cv::Mat Lena_salt_noise2 = Lena_salt.clone();
	cv::Mat Lena_gauss_noise2 = Lena_gauss.clone();
	// 存放高斯滤波后的结果
	cv::Mat gaussianbulr_5, gaussianbulr_9;
	cv::Mat gaussianbulr_5salt, gaussianbulr_9salt;
	cv::Mat gaussianbulr_5gauss, gaussianbulr_9gauss;
	// 进行高斯滤波
	cv::GaussianBlur(Lena_3, gaussianbulr_5, cv::Size(5, 5), 10, 20);
	cv::GaussianBlur(Lena_3, gaussianbulr_9, cv::Size(9, 9), 10, 20);
	cv::GaussianBlur(Lena_salt_noise2, gaussianbulr_5salt, cv::Size(5, 5), 10, 20);
	cv::GaussianBlur(Lena_salt_noise2, gaussianbulr_9salt, cv::Size(9, 9), 10, 20);
	cv::GaussianBlur(Lena_gauss_noise2, gaussianbulr_5gauss, cv::Size(5, 5), 10, 20);
	cv::GaussianBlur(Lena_gauss_noise2, gaussianbulr_9gauss, cv::Size(9, 9), 10, 20);
	// 显示处理结果
	cv::imshow("Lean6", Lena_3);
	cv::imshow("gaussianbulr_5", gaussianbulr_5);
	cv::imshow("gaussianbulr_9", gaussianbulr_9);

	cv::imshow("Lena_salt_noise2", Lena_salt_noise2);
	cv::imshow("gaussianbulr_5salt", gaussianbulr_5salt);
	cv::imshow("gaussianbulr_9salt", gaussianbulr_9salt);

	cv::imshow("Lena_gauss_noise2", Lena_gauss_noise2); 
	cv::imshow("gaussianbulr_5gauss", gaussianbulr_5gauss);
	cv::imshow("gaussianbulr_9gauss", gaussianbulr_9gauss);

	cv::waitKey(0);
	cv::destroyAllWindows();

	// 可分离滤波（自定义滤波器）
	/*
	cv::sepFilter2D(InputArray src, 					输入图像
					OutputArray dst, 					输出图像
					int ddepth,							输出图像的数据类型
                    InputArray kernelX, 				X方向的滤波器
					InputArray kernelY,					Y方向的滤波器
                    Point anchor = Point(-1,-1),		滤波器的锚点
                    double delta = 0, 					偏置				
					int borderType = BORDER_DEFAULT 	像素外推法选择标志
					);
	*/
	// 待卷积矩阵（再次使用该矩阵） 
	float points[25] = { 1,2,3,4,5,
		6,7,8,9,10,
		11,12,13,14,15,
		16,17,18,19,20,
		21,22,23,24,25 };
	cv::Mat mat3(5, 5, CV_32FC1, points);
	
	//X方向、Y方向和联合滤波器的构建
	cv::Mat a = (cv::Mat_<float>(3, 1) << -1, 3, -1);
	cv::Mat b = a.reshape(1, 1);
	cv::Mat ab = a*b;
	/*
	a=
		[-1;
 		  3;
 		 -1]
	b=
		[-1, 3, -1]
	ab=
		[ 1, -3,  1;
 		 -3,  9, -3;
 		  1, -3,  1]
	*/

	//验证高斯滤波的可分离性
	cv::Mat gaussY = cv::getGaussianKernel(3, 1);
	cv::Mat gaussX = gaussY.reshape(1, 1);
	/*
	gaussY=
			[0.274068619061197;
 			 0.451862761877606;
 			 0.274068619061197]
	gaussX=
			[0.274068619061197, 0.451862761877606, 0.274068619061197]
	*/
	cv::Mat gaussData, gaussDataXY, gaussDataXX;
	cv::GaussianBlur(mat3, gaussData, cv::Size(3, 3), 1, 1, cv::BORDER_CONSTANT);
	cv::sepFilter2D(mat3, gaussDataXY, -1, gaussX, gaussY, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	cv::sepFilter2D(mat3, gaussDataXX, -1, gaussX, gaussX, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	//输入两种高斯滤波的计算结果
	std::cout << "gaussData=" << std::endl
		<< gaussData << std::endl;
	std::cout << "gaussDataXY=" << std::endl
		<< gaussDataXY << std::endl;
	std::cout << "gaussDataXX=" << std::endl
		<< gaussDataXX << std::endl;
	std::cout << std::endl;
	// filter2D()需要确定滤波器的尺寸是1xK(X方向滤波)还是Kx1(Y方向滤波)
	// sepFilter2D()不需要确定滤波器滤波方向

	//线性滤波的可分离性
	cv::Mat dataYX, dataY, dataXY, dataXY_sep;
	cv::filter2D(mat3, dataY, -1, a, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	cv::filter2D(dataY, dataYX, -1, b, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	cv::filter2D(mat3, dataXY, -1, ab, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	cv::sepFilter2D(mat3, dataXY_sep, -1, a, b, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	//输出分离滤波和联合滤波的计算结果
	std::cout << "dataY=" << std::endl
		<< dataY << std::endl;
	std::cout << "dataYX=" << std::endl
		<< dataYX << std::endl;
	std::cout << "dataXY=" << std::endl
		<< dataXY << std::endl;
	std::cout << "dataXY_sep=" << std::endl
		<< dataXY_sep << std::endl;
	std::cout << std::endl;

	//对图像的分离操作
	cv::Mat Lena_4 = Lena.clone();
	cv::Mat LenaYX, LenaY, LenaXY;
	cv::filter2D(Lena_4, LenaY, -1, a, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	cv::filter2D(LenaY, LenaYX, -1, b, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	cv::filter2D(Lena_4, LenaXY, -1, ab, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	cv::imshow("Lena_4", Lena_4);
	cv::imshow("LenaY", LenaY);
	cv::imshow("LenaYX", LenaYX);
	cv::imshow("LenaXY", LenaXY);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/* 非线性滤波 */
	// 中值滤波（对斑点噪声和椒盐噪声的处理具有较好的效果）
	// 优点：对脉冲干扰信号和图像扫描噪声的处理效果好、保护边缘信息
	// 缺点：尺寸不能太大(太大会造成模糊,丢失边缘信息)，消耗时间长
	/*
	中值滤波
	cv::medianBlur( InputArray src, 	输入图像
					OutputArray dst, 	输出图像
					int ksize 			滤波器尺寸
					);
	*/
	cv::Mat medianblur_3, medianblur_9, medianblur_gray_3, medianblur_gray_9;
	// 分别对含有椒盐噪声的彩色和灰度图像进行滤波
	cv::medianBlur(Lena_salt, medianblur_3, 3);
	cv::medianBlur(Lena_salt, medianblur_9, 9);

	cv::medianBlur(Lena_gray_salt, medianblur_gray_3, 3);
	cv::medianBlur(Lena_gray_salt, medianblur_gray_9, 9);

	cv::imshow("Lena_salt1", Lena_salt);
	cv::imshow("medianblur_3", medianblur_3);
	cv::imshow("medianblur_9", medianblur_9);
	cv::imshow("Lena_gray_salt1", Lena_gray_salt);
	cv::imshow("medianblur_gray_3", medianblur_gray_3);
	cv::imshow("medianblur_gray_9", medianblur_gray_9);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// 双边滤波
	/*
	双边滤波
	cv::bilateralFilter(InputArray src, 					输入图像
						OutputArray dst, 					输出图像
						int d,								滤波过程中每个像素领域的直径
                        double sigmaColor, 					颜色空间滤波器的标准差值
						double sigmaSpace,					空间坐标中滤波器的标准差值
                        int borderType = BORDER_DEFAULT 	像素外推法选择标志
						);
	*/
	cv::Mat bilateralresult1, bilateralresult2, bilateralresult3, bilateralresult4;
	//验证不同滤波器直径的滤波效果
	cv::bilateralFilter(Lena, bilateralresult1, 9, 50, 25 / 2);
	cv::bilateralFilter(Lena, bilateralresult2, 25, 50, 25 / 2);
	//验证不同标准差值的滤波效果
	cv::bilateralFilter(Lena, bilateralresult3, 9, 9, 9);
	cv::bilateralFilter(Lena, bilateralresult4, 9, 200, 200);
	//显示原图
	cv::imshow("Lena_5", Lena);
	//不同直径滤波结果
	cv::imshow("bilateralresult1", bilateralresult1);
	cv::imshow("bilateralresult2", bilateralresult2);
	//不同标准差值滤波结果
	cv::imshow("bilateralresult3 ", bilateralresult3);
	cv::imshow("bilateralresult4", bilateralresult4);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/* 图像的边缘检测 */
	/*
	矩阵绝对值(dst(I) = |src(I)*alpha+beta| )
	cv::convertScaleAbs(InputArray src, 	输入图像
						OutputArray dst,	输出图像
                        double alpha = 1, 	缩放因子
						double beta = 0 	偏置
						);
	
	（所有算子由于存在负数，导致结果存在负数，不在原始图像的CV_8U的数据类型内，因此滤波后图像的数据类型应该改为CV_16S）
	Soble算子
	cv::Sobel( 	InputArray src, 					输入图像
				OutputArray dst, 					输出图像
				int ddepth,							输出图像的数据类型
                int dx, 							X方向的差分阶数
				int dy, 							Y方向的差分阶数
				int ksize = 3,						Sobel算子的尺寸
                double scale = 1, 					对导数计算结果的缩放因子
				double delta = 0,					偏置
                int borderType = BORDER_DEFAULT 	像素外推法选择标志
				);
	
	Scharr算子(默认的滤波器尺寸为3x3，并且无法修改)
	cv::Scharr( InputArray src, 					输入图像
				OutputArray dst, 					输出图像
				int ddepth,							输出图像的数据类型
                int dx, 							X方向的差分阶数
				int dy, 							Y方向的差分阶数
				double scale = 1, 					对导数计算结果的缩放因子
				double delta = 0,					偏置
                int borderType = BORDER_DEFAULT 	像素外推法选择标志
				);

	得到边缘检测算子（Soble算子和Scharr算子内部引用此函数）
	cv::getDerivKernels(OutputArray kx, 			行滤波器系数的输出矩阵,尺寸Kx1
						OutputArray ky,				列滤波器系数的输出矩阵,尺寸Kx1
                        int dx, 					X方向导数的阶次
						int dy, 					Y方向导数的阶次
						int ksize,					滤波器的大小
                        bool normalize = false, 	是否对滤波器系数进行归一化的标志
						int ktype = CV_32F 			滤波器系数类型
						);
	
	Laplacian算子（无方向性，只需要一次边缘检测，是一个二阶导数算子，但对噪声比较敏感）
	cv::Laplacian( 	InputArray src, 				输入图像
					OutputArray dst, 				输出图像
					int ddepth,						输出图像的数据类型
                    int ksize = 1, 					滤波器的大小
					double scale = 1, 				对导数计算结果的缩放因子
					double delta = 0,				偏置
                    int borderType = BORDER_DEFAULT 像素外推法选择标志
					);

	Canny算法（不容易受到噪声的影响，能识别图像中的弱边缘和强边缘，并结合强弱边缘的位置关系，综合给出图像整体的边缘信息）
	cv::Canny( 	InputArray image, 			输入图像
				OutputArray edges,			输出图像
                double threshold1, 			第一个滞后阈值
				double threshold2,			第二个滞后阈值
                int apertureSize = 3, 		Sobel算子的直径
				bool L2gradient = false 	计算图像梯度幅值方法的标志
				);
	*/
	// 自生成边缘检测滤波器
	cv::Mat kernel1 = (cv::Mat_<float>(1, 2) << 1, -1);			// X方向边缘检测滤波器
	cv::Mat kernel2 = (cv::Mat_<float>(1, 3) << 0.5, 0, -0.5);	// X方向边缘检测滤波器
	cv::Mat kernel3 = (cv::Mat_<float>(3, 1) << 0.5, 0, -0.5);	// Y方向边缘检测滤波器
	// 检测图像边缘
	cv::Mat edge1, edge2, edge3, edge4;
	// 以[1 -1]检测水平方向边缘
	cv::filter2D(Lena, edge1, CV_16S, kernel1);
	cv::convertScaleAbs(edge1, edge1);
	// 以[0.5 0 -0.5]检测水平方向边缘
	cv::filter2D(Lena, edge2, CV_16S, kernel2);
	cv::convertScaleAbs(edge2, edge2);
	// 以[0.5 0 -0.5]检测垂直方向边缘
	cv::filter2D(Lena, edge3, CV_16S, kernel3);
	cv::convertScaleAbs(edge3, edge3);
	// 综合整幅图像的边缘
	edge4 = edge2 + edge3;
	// 显示结果
	cv::imshow("edge1", edge1);
	cv::imshow("edge2", edge2);
	cv::imshow("edge3", edge3);
	cv::imshow("edge4", edge4);
	cv::waitKey(0);
	cv::destroyAllWindows();
	
	// Sobel算子
	cv::Mat SobelX, SobelY, SobelXY;
	// X方向一阶边缘检测
	cv::Sobel(Lena, SobelX, CV_16S, 1, 0, 3);
	cv::convertScaleAbs(SobelX, SobelX);
	// Y方向一阶边缘检测
	cv::Sobel(Lena, SobelY, CV_16S, 0, 1, 3);
	cv::convertScaleAbs(SobelY, SobelY);
	// 整幅图像的一阶边缘检测
	SobelXY = SobelX + SobelY;
	// 显示结果
	cv::imshow("SobelX", SobelX);
	cv::imshow("SobelY", SobelY);
	cv::imshow("SobelXY", SobelXY);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// Scharr算子s
	cv::Mat ScharrX, ScharrY, ScharrXY;
	// X方向一阶边缘检测
	cv::Scharr(Lena, ScharrX, CV_16S, 1, 0);
	cv::convertScaleAbs(ScharrX, ScharrX);
	// Y方向一阶边缘检测
	cv::Scharr(Lena, ScharrY, CV_16S, 0, 1);
	cv::convertScaleAbs(ScharrY, ScharrY);
	// 整幅图像的一阶边缘检测
	ScharrXY = ScharrX + ScharrY;
	// 显示结果
	cv::imshow("ScharrX", ScharrX);
	cv::imshow("ScharrY", ScharrY);
	cv::imshow("ScharrXY", ScharrXY);
	cv::waitKey(0);
	cv::destroyAllWindows();

	cv::Mat sobel_x1, sobel_y1, sobel_x2, sobel_y2, sobel_x3, sobel_y3;  //存放分离的Sobel算子
	cv::Mat scharr_x, scharr_y;  //存放分离的Scharr算子
	cv::Mat sobelX1, sobelX2, sobelX3, scharrX;  //存放最终算子

	//一阶X方向Sobel算子
	cv::getDerivKernels(sobel_x1, sobel_y1, 1, 0, 3);
	sobel_x1 = sobel_x1.reshape(CV_8U, 1);
	sobelX1 = sobel_y1*sobel_x1;  //计算滤波器

	//二阶X方向Sobel算子
	cv::getDerivKernels(sobel_x2, sobel_y2, 2, 0, 5);
	sobel_x2 = sobel_x2.reshape(CV_8U, 1);
	sobelX2 = sobel_y2*sobel_x2;  //计算滤波器

	//三阶X方向Sobel算子
	cv::getDerivKernels(sobel_x3, sobel_y3, 3, 0, 7);
	sobel_x3 = sobel_x3.reshape(CV_8U, 1);
	sobelX3 = sobel_y3*sobel_x3;  //计算滤波器

	//X方向Scharr算子
	cv::getDerivKernels(scharr_x, scharr_y, 1, 0, cv::FILTER_SCHARR);
	scharr_x = scharr_x.reshape(CV_8U, 1);
	scharrX = scharr_y*scharr_x;  //计算滤波器

	//输出结果
	std::cout << "X方向一阶Sobel算子:" << std::endl 
			<< sobelX1 << std::endl;
	std::cout << "X方向二阶Sobel算子:" << std::endl 
			<< sobelX2 << std::endl;
	std::cout << "X方向三阶Sobel算子:" << std::endl 
			<< sobelX3 << std::endl;
	std::cout << "X方向Scharr算子:" << std::endl 
			<< scharrX << std::endl;
	std::cout << std::endl;

	// Laplacian算子
	cv::Mat Laplacianresult, Laplacianresult_g, Laplacianresult_G;
	// 未滤波提取Laplacian边缘
	cv::Laplacian(Lena, Laplacianresult, CV_16S, 3, 1, 0);
	cv::convertScaleAbs(Laplacianresult, Laplacianresult);
	// 滤波后提取Laplacian边缘
	cv::GaussianBlur(Lena, Laplacianresult_g, cv::Size(3, 3), 5, 0);  //高斯滤波
	cv::Laplacian(Laplacianresult_g, Laplacianresult_G, CV_16S, 3, 1, 0);
	cv::convertScaleAbs(Laplacianresult_G, Laplacianresult_G);
	// 显示结果
	cv::imshow("result", Laplacianresult);
	cv::imshow("result_G", Laplacianresult_G);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// Canny算子
	cv::Mat resultHigh, resultLow, resultG;
	// 高阈值检测图像边缘
	cv::Canny(Lena_gray, resultHigh, 100, 200, 3);
	// 低阈值检测图像边缘
	cv::Canny(Lena_gray, resultLow, 20, 40, 3);
	// 高斯模糊后提取图像边缘
	cv::GaussianBlur(Lena_gray, resultG, cv::Size(3, 3), 5);
	cv::Canny(resultG, resultG, 100, 200, 3);
	// 显示结果
	cv::imshow("Lena", Lena);
	cv::imshow("resultHigh", resultHigh);
	cv::imshow("resultLow", resultLow);
	cv::imshow("resultG", resultG);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}