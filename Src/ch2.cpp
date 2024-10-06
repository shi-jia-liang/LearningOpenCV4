#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
	// 第一种创建Mat类
	cv::Mat a1; // 创建一个名为a1的矩阵头
	// 第二种创建Mat类
	cv::Mat a2 = cv::Mat_<double>(3, 3); // 创建一个3x3的矩阵存放double类型数据
	// 第三种创建Mat类
	/* 
	数据类型	具体类型				取值范围
	CV_8U	8位无符号整数(uchar)	0~255
	CV_8S	8位符号整数(char)		-128~127
	CV_16U	16位无符号整数(ushort)	0~65535
	CV_16S	16位符号整数(short)		-32768~32767
	CV_32S	32位符号整数(int)		-2147483648~2147483647
	CV_32F	32位浮点整数(float)		-FLT_MAX~FLT_MAX
	CV_64F	64位浮点整数(double)	-DBL_MAX~DBL_MAX
	*/
	cv::Mat a3(3, 3, CV_8UC1); // 创建一个3x3的8位无符号整数的单通道矩阵
	cv::Mat a4(3, 3, CV_8U);   // 创建一个3x3的8位无符号整数的单通道矩阵，C1标识可省略
	cv::Mat a5(3, 3, CV_64F);  // 创建一个3x3的矩阵存放double类型数据

	// Mat类的构造	具体可参考源码，通过cv::Mat按F12“转到定义”找到，共18种构造方式
	/*
	利用矩阵尺寸和类型参数构造Mat类
	cv::Mat::Mat(	int rows,		矩阵的行数
					int cols, 		矩阵的列数
					int type		矩阵中保存中的数据类型
					);		

	利用Size()结构构造Mat类
	cv::Mat::Mat(	Size size(),
					int type		矩阵中保存中的数据类型
					);		
	*/

	// Mat类的赋值
	// (1)构造时赋值
	cv::Mat b1(3, 3, CV_8UC3, cv::Scalar(255, 255, 255));
	// (2)枚举法赋值
	cv::Mat b2 = (cv::Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
	// (3)循环法赋值
	cv::Mat b3 = cv::Mat_<int>(3, 3);
	for (int i = 1; i < b3.rows; i++)
	{
		for (int j = 1; j < b3.cols; j++)
		{
			b3.at<int>(i, j) = i + j;
		}
	}
	// (4)类方法赋值
	cv::Mat b4 = cv::Mat::eye(3, 3, CV_8UC1);					  // 构建一个单位矩阵
	cv::Mat b6 = cv::Mat::diag((cv::Mat_<int>(1, 3) << 1, 2, 3)); // 构建一个对角矩阵，其参数必须是Mat类型的一维变量，用来存放对角元素的数值
	cv::Mat b7 = cv::Mat::ones(3, 3, CV_8UC1);					  // 构建一个全为1的矩阵
	cv::Mat b8 = cv::Mat::zeros(3, 3, CV_8UC1);					  // 构建一个零矩阵
	// (5)利用数据进行赋值
	float b[8] = {1, 2, 3, 5, 6, 7, 8};
	cv::Mat b9 = cv::Mat(2, 2, CV_32FC2, b);
	cv::Mat b10 = cv::Mat(2, 4, CV_32FC1, b);

	// Mat类支持的运算
	cv::Mat c1 = (cv::Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
	cv::Mat c2 = (cv::Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
	cv::Mat c3 = (cv::Mat_<double>(3, 3) << 1.0, 2.1, 3.2, 4.0, 5.1, 6.2, 2, 2, 2);
	cv::Mat c4 = (cv::Mat_<double>(3, 3) << 1.0, 2.1, 3.2, 4.0, 5.1, 6.2, 2, 2, 2);
	cv::Mat add, minus, minus1, times, divide;
	// 加减运算中，保证两个矩阵中的数据类型时相同的
	add = c1 + c2;
	minus = c3 - c4;
	// 常数与Mat类变量的加减运算
	minus1 = c3 - 1.0;	// 此表示Mat类变量中的所有元素减去这个常数
	// 常数与Mat类变量的乘除运算，运算结果的数据类型保留Mat类变量的数据类型
	times = 2.0 * c1;	// 结果仍为int类型Mat类
	divide = c4 / 2.0;	// 结果仍为double类型Mat类
	// 在对图像进行卷积运算时，需要两个矩阵进行乘法运算
	cv::Mat c5,c6; double k;
	c5 = c3*c4;			// 矩阵乘法，c_ij = a_i1*b_1j + a_i2*b_2j + a_i3*b_3j，要求第一个Mat类矩阵的列数必须与第二个Mat类矩阵的行数相同，而且该运算要求Mat类中的数据类型必须是CV_32FC1、CV_32FC2、CV_64FC1、CV_64FC2中的四个之一
	k = c1.dot(c2);		// 矩阵内积，k = a_1*b_1 + a_2*b_2 + …… + a_9*b_9，结果为double类型的变量，该运算的目的是求取一个行向量和一个列向量的点乘，要求输入的Mat类矩阵必须有相同的元素数目，无论输出的Mat类矩阵的维数是多少，都会将两个Mat类矩阵扩展成一个行向量和一个列向量
	c6 = c1.mul(c2);	// 矩阵对应位的乘积，c_ij = a_ij*b_ij,只要求两个Mat类矩阵的行列数和数据类型相同即可，数据类型可以是CV_8U、CV_8S、CV_16U等等
	// Mat类元素的读取
	/*
	Mat类矩阵常用的属性
	rows		矩阵的行数
	cols		矩阵的列数
	step		以字节为单位的矩阵的有效宽度 = elemSize() * cols
	elemSize()	每个元素的字节数 = 数据类型位宽 / 8 * channels()
	total()		矩阵中元素的个数 = rows * cols
	channels()	矩阵的通道数
	*/
	// (1)通过at方法读取Mat类矩阵中的元素
	// (2)通过指针ptr读取Mat类矩阵中的元素
	// (3)通过迭代器访问Mat类矩阵中的元素
	// (4)通过矩阵元素的地址定位方式访问元素

	// 图像的读取与显示
	/*
	cv::imread( const String& filename,				需要读取的图像文件名称
	 			int flags = IMREAD_COLOR 			读取图像形式的标志，常见的图像形式参数：	IMREAD_GRAYSCALE(单通道灰色图像后读取)、
																							IMREAD_COLOR(3通道BGR彩色图像)
				);
	cv::namedWindow(const String& winname, 			窗口名称
					int flags = WINDOW_AUTOSIZE		窗口属性设置标志，常见的窗口属性标志参数：	WINDOW_NORMAL(显示图像后，允许用户随意调整窗口大小)、
																							WINDOW_AUTOSIZE(根据图像大小显示窗口，不允许用户随意调整大小)、
																							WINDOW_KEEPRATIO(保持图像的比例)、
																							WINDOW_FULLSCREEN(全屏显示窗口)
					);
	cv::imshow(	const String& winname,				窗口名称
				InputArray mat						要显示的图像矩阵
				);
	cv::VideoCapture(	const String& filename, 	需要读取的视频文件名称
						int apiPreference = CAP_ANY	读取数据时设置的属性
						);
	cv::imwrite(const String& filename,								保存图像的地址和文件名，包括图像格式
				InputArray img,										将要保存的Mat类矩阵变量
              	const std::vector<int>& params = std::vector<int>()	保存图片格式属性设置标志
				);
	cv::VideoWriter(const String& filename, 		保存视频的地址和文件名，包含视频格式
					int fourcc, 					压缩帧的4字符编解码器代码
					double fps,						保存视频的帧率
                	Size frameSize, 				视频帧的尺寸
					bool isColor = true				保存视频是否为彩色视频
					);
	*/
	return 0;
}