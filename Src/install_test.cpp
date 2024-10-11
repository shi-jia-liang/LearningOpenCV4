#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

/*
	使用命名空间：
		1.避免命名冲突：
			当你有多个库或模块，并且它们可能包含相同名称的函数、变量或类型时，使用命名空间可以避免命名冲突。
		2.组织代码：
			命名空间可以用来组织代码，使其更具结构性和可读性。例如，你可以将所有与数学相关的函数放在一个名为 Math 的命名空间中。
		3.模块化设计：
			大型项目通常会使用命名空间来模块化设计，使得代码结构更加清晰。
	使用类：
		1.封装数据和函数：
			类允许你将数据和操作这些数据的函数封装在一起，形成对象。这是面向对象编程（OOP）的核心概念。
		2.定义类型：
			类可以定义新的类型，这些类型可以包含属性和方法。
		3.多态性和继承：
			类支持多态性和继承，这使得代码可以更加灵活和可扩展。
*/

int main()	
{
	// 测试程序
	cout << "hello world!" << endl;
	// 读取图片
	cv::Mat img = cv::imread("../Img/Lena_origin.jpg");
	cout << "Width : " << img.size().width << endl;
	cout << "Height : " << img.size().height << endl;
	cout << "Channels : " << img.channels() << endl;
	// 裁剪图片
	cv::Mat cropped_img = img(cv::Range(0, 512), cv::Range(250, 512+250));
	// 展示图片、保存图片
	cv::imshow("test", img);
	cv::waitKey(0);
	cv::imshow("test", cropped_img);
	cv::imwrite("../Img/Lena.jpg", cropped_img);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}
