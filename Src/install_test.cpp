#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

int main()
{
	cv::Mat img;
	cout << "hello world!" << endl;
	img = cv::imread("D:/Projects/opencv4_test/Img/Lena_origin.png");
	cv::imshow("test", img);
	cv::waitKey(0);
	return 0;
}
