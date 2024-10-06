#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

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
