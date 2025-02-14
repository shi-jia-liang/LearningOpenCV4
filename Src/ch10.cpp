#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

//使用initUndistortRectifyMap()函数和remap()函数校正图像
void initUndistAndRemap(
	std::vector<cv::Mat> imgs,  //所有原图像向量
	cv::Mat cameraMatrix,  //计算得到的相机内参
	cv::Mat distCoeffs,    //计算得到的相机畸变系数
	cv::Size imageSize,    //图像的尺寸
	std::vector<cv::Mat> &undistImgs)  //校正后的输出图像
{
	//计算映射坐标矩阵
	cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat mapx = cv::Mat(imageSize, CV_32FC1);
	cv::Mat mapy = cv::Mat(imageSize, CV_32FC1);
	initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, imageSize, CV_32FC1, mapx, mapy);

	//校正图像
	for (int i = 0; i < imgs.size(); i++)
	{
		cv::Mat undistImg;
		remap(imgs[i], undistImg, mapx, mapy, cv::INTER_LINEAR);
		undistImgs.push_back(undistImg);
	}
}

//用undistort()函数直接计算校正图像
void undist(
	std::vector<cv::Mat> imgs,   //所有原图像向量
	cv::Mat cameraMatrix,   //计算得到的相机内参
	cv::Mat distCoeffs,     //计算得到的相机畸变系数
	std::vector<cv::Mat> &undistImgs)  //校正后的输出图像
{
	for (int i = 0; i < imgs.size(); i++)
	{
		cv::Mat undistImg;
		cv::undistort(imgs[i], undistImg, cameraMatrix, distCoeffs);
		undistImgs.push_back(undistImg);
	}
}

//检测棋盘格内角点在图像中坐标的函数
void getImgsPoints(std::vector<cv::Mat> imgs, std::vector<std::vector<cv::Point2f>> &Points, cv::Size boardSize)
{
	for (int i = 0; i < imgs.size(); i++)
	{
		cv::Mat img1 = imgs[i];
		cv::Mat gray1;
		cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
		std::vector<cv::Point2f> img1_points;
		findChessboardCorners(gray1, boardSize, img1_points);  //计算方格标定板角点
		find4QuadCornerSubpix(gray1, img1_points, cv::Size(5, 5));  //细化方格标定板角点坐标
		Points.push_back(img1_points);
	}
}

int main(){
	/* 单目相机 */
	/*
	非齐次坐标想齐次坐标转换
	cv::convertPointsToHomogeneous(	InputArray src, 		// 输入非齐次坐标
									OutputArray dst 		// 输出齐次坐标
									);  

	齐次坐标向非齐次坐标转换
	cv::convertPointsFromHomogeneous(	InputArray src,		// 输入齐次坐标
										OutputArray dst 	// 输出非齐次坐标
										); 
	棋盘格内角点检测
	cv::findChessboardCorners(	InputArray image, 													// 输入图像
								Size patternSize, 													// 棋盘格尺寸
								OutputArray corners, 												// 输出角点
								int flags = CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE 	// 标志位
								); 																	// 返回值bool数据类型，表示是否检测到角点的结果，true表示有，false表示无
	
	内角点位置优化
	cv::cornerSubPix(	InputArray image, 			// 输入图像
						InputOutputArray corners, 	// 输入输出角点
						Size winSize, 				// 窗口尺寸
						Size zeroZone, 				// 零区域尺寸
						TermCriteria criteria 		// 终止条件
						); 							// 返回值bool数据类型，表示是否优化角点位置的结果，true表示有，false表示无
	
	圆形网格的圆心检测
	cv::findCirclesGrid(	InputArray image, 														// 输入图像
						Size patternSize, 															// 圆形网格尺寸
						OutputArray centers, 														// 输出圆心
						int flags = CALIB_CB_SYMMETRIC_GRID, 										// 标志位
						const cv::Ptr<FeatureDetector>& blobDetector = cv::Ptr<FeatureDetector>() 	// 特征检测器
						); 																			// 返回值bool数据类型，表示是否检测到圆心的结果，true表示有，false表示无
						
	绘制棋盘格的内角点或者圆形网格的圆心
	cv::drawChessboardCorners(	InputOutputArray image, 	// 输入输出图像
								Size patternSize, 			// 棋盘格尺寸
								InputArray corners, 		// 角点或者圆心
								bool patternWasFound 		// 是否检测到角点或者圆心
								); 							// 返回值bool数据类型，表示是否绘制角点或者圆心的结果，true表示有，false表示无
	
	图像去畸变校正
	cv::undistort(	InputArray src, 						// 输入图像
					OutputArray dst, 						// 输出图像
					InputArray cameraMatrix, 				// 相机内参矩阵
					InputArray distCoeffs, 					// 畸变系数
					InputArray newCameraMatrix = noArray() 	// 新的相机内参矩阵
					); 										// 返回值bool数据类型，表示是否去畸变校正的结果，true表示有，false表示无

	单目相机空间点向图像投影
	cv::projectPoints(	InputArray objectPoints, 			// 世界坐标系中的三维点
						InputArray rvec, 					// 旋转向量
						InputArray tvec, 					// 平移向量
						InputArray cameraMatrix, 			// 相机内参矩阵
						InputArray distCoeffs, 				// 畸变系数
						OutputArray imagePoints, 			// 输出图像坐标系中的二维点
						OutputArray jacobian = noArray() 	// 输出雅克比矩阵
						); 									// 返回值bool数据类型，表示是否投影的结果，true表示有，false表示无

	计算位姿关系
	cv::solvePnP(	InputArray objectPoints, 			// 世界坐标系中的三维点
					InputArray imagePoints, 			// 图像坐标系中的二维点
					InputArray cameraMatrix, 			// 相机内参矩阵
					InputArray distCoeffs, 				// 畸变系数
					OutputArray rvec, 					// 旋转向量
					OutputArray tvec, 					// 平移向量
					bool useExtrinsicGuess = false, 	// 是否使用外参猜测
					int flags = SOLVEPNP_ITERATIVE 		// 标志位
					); 									// 返回值bool数据类型，表示是否计算位姿关系的结果，true表示有，false表示无

	旋转向量与旋转矩阵相互转换
	cv::Rodrigues(	InputArray src, 		// 输入旋转向量
					OutputArray dst 		// 输出旋转矩阵
					); 						// 返回值bool数据类型，表示是否转换的结果，true表示有，false表示无
	*/
	//设置两个三维坐标
	std::vector<cv::Point3f> points3;
	points3.push_back(cv::Point3f(3, 6,1.5));
	points3.push_back(cv::Point3f(23, 32, 1));

	//非齐次坐标转齐次坐标
	cv::Mat points4;
	convertPointsToHomogeneous(points3, points4);

	//齐次坐标转非齐次坐标
	std::vector<cv::Point2f> points2;
	convertPointsFromHomogeneous(points3, points2);

	std::cout << "***********齐次坐标转非齐次坐标*************" << std::endl;
	for (int i = 0; i < points3.size(); i++)
	{
		std::cout << "齐次坐标：" << points3[i];
		std::cout<< "   非齐次坐标：" << points2[i] << std::endl;
	}

	std::cout << "***********非齐次坐标转齐次坐标*************" << std::endl;
	for (int i = 0; i < points3.size(); i++)
	{
		std::cout << "齐次坐标：" << points3[i];
		std::cout << "   非齐次坐标：" << points4.at<cv::Vec4f>(i, 0) << std::endl;
	}
	cv::waitKey(0);

	cv::Mat cheess = cv::imread("../Img/cal/left01.jpg");
	cv::Mat circle = cv::imread("../Img/circle.png");
	if (!(cheess.data && circle.data))
	{
		std::cout << "读取图像错误，请确认图像文件是否正确" << std::endl;
		return -1;
	}
	cv::Mat cheessgray, circlegray;
	cvtColor(cheess, cheessgray, cv::COLOR_BGR2GRAY);
	cvtColor(circle, circlegray, cv::COLOR_BGR2GRAY);

	// 定义数目尺寸
	cv::Size cheess_size = cv::Size(9, 6);   // 方格标定板内角点数目（行，列）
	cv::Size circle_size = cv::Size(7, 7);   // 圆形标定板圆心数目（行，列）

	// 检测角点
	std::vector<cv::Point2f> cheess_points, circle_points;  
	cv::findChessboardCorners(cheessgray, cheess_size, cheess_points);  // 计算方格标定板角点
	cv::findCirclesGrid(circlegray, circle_size, circle_points);  // 计算圆形标定板检点

	// 细化角点坐标
	cv::find4QuadCornerSubpix(cheessgray, cheess_points, cv::Size(5, 5));  // 细化方格标定板角点坐标
	cv::find4QuadCornerSubpix(circlegray, circle_points, cv::Size(5, 5));  // 细化圆形标定板角点坐标

	// 绘制角点检测结果
	cv::drawChessboardCorners(cheess, cheess_size, cheess_points, true);
	cv::drawChessboardCorners(circle, circle_size, circle_points, true);

	// 显示结果
	cv::imshow("方形标定板角点检测结果", cheess);
	cv::imshow("圆形标定板角点检测结果", circle);
	cv::waitKey(0);
	cv::destroyAllWindows();

	//读取所有图像
	std::vector<cv::Mat> Calibrateimgs;
	std::string imageName;
	std::ifstream  fin("../Img/cal/steroCalibDataL.txt");	// 读取左相机图像文件名,文件中存放图像文件名(请注意文件路径,相对于该文件的相对路径)
	while (getline(fin,imageName))
	{
		cv::Mat img = cv::imread(imageName);
		Calibrateimgs.push_back(img);
	}

	cv::Size board_size = cv::Size(9, 6);  // 方格标定板内角点数目（行，列）
	std::vector<std::vector<cv::Point2f>> CalibrateimgsPoints;
	for (int i = 0; i < Calibrateimgs.size(); i++)
	{
		cv::Mat img1 = Calibrateimgs[i];
		cv::Mat gray1;
		cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
		std::vector<cv::Point2f> img1_points;
		findChessboardCorners(gray1, board_size, img1_points);  // 计算方格标定板角点
		find4QuadCornerSubpix(gray1, img1_points, cv::Size(5, 5));  // 细化方格标定板角点坐标
		CalibrateimgsPoints.push_back(img1_points);
	}

	// 生成棋盘格每个内角点的空间三维坐标
	cv::Size boardSize = cv::Size(9, 6);	 // 棋盘格内角点数目（行，列）
	cv::Size squareSize = cv::Size(10, 10);  // 棋盘格每个方格的真实尺寸
	std::vector<std::vector<cv::Point3f>> objectPoints;
	for (int i = 0; i < CalibrateimgsPoints.size(); i++)
	{
		std::vector<cv::Point3f> tempPointSet;
		for (int j = 0; j < board_size.height; j++)
		{
			for (int k = 0; k < board_size.width; k++)
			{
				cv::Point3f realPoint;
				// 假设标定板为世界坐标系的z平面，即z=0
				realPoint.x = j * squareSize.width;
				realPoint.y = k * squareSize.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		objectPoints.push_back(tempPointSet);
	}

	/* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
	std::vector<int> point_number;
	for (int i = 0; i<CalibrateimgsPoints.size(); i++)
	{
		point_number.push_back(board_size.width*board_size.height);
	}

	// 图像尺寸
	cv::Size imageSize;
	imageSize.width = Calibrateimgs[0].cols;
	imageSize.height = Calibrateimgs[0].rows;

	cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0));  // 摄像机内参数矩阵
	cv::Mat distCoeffs = cv::Mat(1, 5, CV_32FC1, cv::Scalar::all(0));  // 摄像机的5个畸变系数：k1,k2,p1,p2,k3
	std::vector<cv::Mat> rvecs;  // 每幅图像的旋转向量
	std::vector<cv::Mat> tvecs;  // 每张图像的平移向量
	cv::calibrateCamera(objectPoints, CalibrateimgsPoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, 0);
	std::cout << "相机的内参矩阵=" << std::endl << cameraMatrix << std::endl;
	std::cout << "相机畸变系数" << distCoeffs << std::endl;
	for (int i = 0; i < rvecs.size(); i++)
	{
		std::cout << "图像 " << i + 1 << " 的旋转向量: " << rvecs[i].t() << std::endl;	// t()表示转置
		std::cout << "图像 " << i + 1 << " 的平移向量: " << tvecs[i].t() << std::endl;	// t()表示转置
	}
	cv::waitKey(0);

	// // 输入前文计算得到的内参矩阵
	// cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << 532.740145, 		  , 334.856092, 
	// 	                                     				  	 0, 532.110552, 234.098277, 
	// 	                                     				  	 0, 		 0, 		 1);
	// // 输入前文计算得到的畸变矩阵
	// cv::Mat distCoeffs = (cv::Mat_<float>(1, 5) << -0.283487, 0.078951, 0.001817, -0.002869, 0.109727);
	// // 输入前文计算得到第一张图像的旋转向量
	// cv::Mat rvec = (cv::Mat_<float>(3, 1) << -1.978694, -2.003831, 0.126608);
	// // 输入前文计算得到第一张图像的平移向量
	// cv::Mat tvec = (cv::Mat_<float>(3, 1) << -27.678758, -43.016370, 159.217514);

	// 校正后的图像
	std::vector<cv::Mat> undistImgs;

	// 使用initUndistortRectifyMap()函数和remap()函数校正图像
	initUndistAndRemap(Calibrateimgs, cameraMatrix, distCoeffs, imageSize, undistImgs);

	// 用undistort()函数直接计算校正图像，下一行代码取消注释即可
	// undist(imgs, cameraMatrix, distCoeffs, undistImgs);

	// 显示校正前后的图像
	for (int i = 0; i < Calibrateimgs.size(); i++)
	{
		std::string windowNumber = std::to_string(i);
		cv::imshow("未校正图像"+ windowNumber, Calibrateimgs[i]);
		cv::imshow("校正后图像"+ windowNumber, undistImgs[i]);
	}
	cv::waitKey(0);
	cv::destroyAllWindows();

	std::vector<cv::Point3f> PointSets;
	for (int j = 0; j < boardSize.height; j++)
	{
		for (int k = 0; k < boardSize.width; k++)
		{
			cv::Point3f realPoint;
			// 假设标定板为世界坐标系的z平面，即z=0
			realPoint.x = j * squareSize.width;
			realPoint.y = k * squareSize.height;
			realPoint.z = 0;
			PointSets.push_back(realPoint);
		}
	}

	// 根据三维坐标和相机与世界坐标系时间的关系估计内角点像素坐标
	std::vector<cv::Point2f> imagePoints;
	projectPoints(PointSets, rvecs[0], tvecs[0], cameraMatrix, distCoeffs, imagePoints);

	/***********计算图像中内角点的真实坐标误差******************/
	std::vector<cv::Point2f> imgPoints;
	cv::findChessboardCorners(cheessgray, boardSize, imgPoints);  // 计算方格标定板角点
	cv::find4QuadCornerSubpix(cheessgray, imgPoints, cv::Size(5, 5));  // 细化方格标定板角点坐标
	//计算估计值和图像中计算的真实时之间的平均误差
	float e = 0;
	for (int i = 0; i < imagePoints.size(); i++)
	{
		float eX = pow(imagePoints[i].x - imgPoints[i].x, 2);
		float eY = pow(imagePoints[i].y - imgPoints[i].y, 2);
		e = e + sqrt(eX + eY);
	}
	e = e / imagePoints.size();
	std::cout << "估计坐标与真实坐标之间的误差:" << e << std::endl;
	cv::waitKey(0);

	//用PnP算法计算旋转和平移量
	cv::Mat rvec, tvec;
	cv::solvePnP(objectPoints[0], CalibrateimgsPoints[0], cameraMatrix, distCoeffs, rvec, tvec);
	std::cout << "世界坐标系变换到相机坐标系的旋转向量：" << rvec << std::endl;
	//旋转向量转换旋转矩阵
	cv::Mat R;
	cv::Rodrigues(rvec, R);
	std::cout << "旋转向量转换成旋转矩阵：" << std::endl << R << std::endl;

	//用PnP+Ransac算法计算旋转向量和平移向量
	cv::Mat rvecRansac, tvecRansac;
	cv::solvePnPRansac(objectPoints[0], CalibrateimgsPoints[0], cameraMatrix, distCoeffs, rvecRansac, tvecRansac);
	cv::Mat RRansac;
	cv::Rodrigues(rvecRansac, RRansac);
	std::cout << "旋转向量转换成旋转矩阵：" << std::endl << RRansac << std::endl;
	cv::waitKey(0);

	/* 双目相机 */
	/*
	双目相机标定
	cv::stereoCalibrate(InputArrayOfArrays objectPoints, 	// 世界坐标系中的三维点
						InputArrayOfArrays imagePoints1, 	// 第一个相机的图像坐标系中的二维点
						InputArrayOfArrays imagePoints2, 	// 第二个相机的图像坐标系中的二维点
						InputOutputArray cameraMatrix1, 	// 第一个相机的内参矩阵
						InputOutputArray distCoeffs1, 		// 第一个相机的畸变系数
						InputOutputArray cameraMatrix2, 	// 第二个相机的内参矩阵
						InputOutputArray distCoeffs2, 		// 第二个相机的畸变系数
						Size imageSize, 					// 图像的尺寸
						InputOutputArray R, 				// 旋转矩阵
						InputOutputArray T, 				// 平移矩阵
						OutputArray E, 						// 本征矩阵
						OutputArray F, 						// 基本矩阵
						TermCriteria criteria, 				// 终止条件
						int flags = CALIB_FIX_INTRINSIC 	// 标志位
						);
	
	双目相机畸变校正					
	cv::stereoRectify(	InputArray cameraMatrix1, 			// 第一个相机的内参矩阵
						InputArray distCoeffs1, 			// 第一个相机的畸变系数
						InputArray cameraMatrix2, 			// 第二个相机的内参矩阵
						InputArray distCoeffs2, 			// 第二个相机的畸变系数
						Size imageSize, 					// 图像的尺寸
						InputArray R, 						// 旋转矩阵
						InputArray T, 						// 平移矩阵
						OutputArray R1, 					// 第一个相机的旋转矩阵
						OutputArray R2, 					// 第二个相机的旋转矩阵
						OutputArray P1, 					// 第一个相机的投影矩阵
						OutputArray P2, 					// 第二个相机的投影矩阵
						OutputArray Q, 						// 重投影矩阵
						int flags = CALIB_ZERO_DISPARITY, 	// 标志位
						double alpha = -1, 					// 图像拉伸系数
						Size newImageSize = Size(), 		// 新图像尺寸
						Rect* validPixROI1 = 0, 			// 第一个相机的有效像素区域
						Rect* validPixROI2 = 0 				// 第二个相机的有效像素区域
						);q
	*/
	std::vector<cv::Mat> imgLs;
	std::vector<cv::Mat> imgRs;
	std::string imgLName;
	std::string imgRName;


	std::ifstream finL("../Img/cal/steroCalibDataL.txt");
	std::ifstream finR("../Img/cal/steroCalibDataR.txt");
	while (getline(finL, imgLName) && getline(finR, imgRName))
	{
		cv::Mat imgL = cv::imread(imgLName);
		cv::Mat imgR = cv::imread(imgRName);
		if (!imgL.data && !imgR.data)
		{
			std::cout << "请确是否输入正确的图像文件" << std::endl;
			return -1;
		}
		imgLs.push_back(imgL);
		imgRs.push_back(imgR);
	}

	// 提取棋盘格内角点在两个相机图像中的坐标
	cv::Size stereoboard_size = cv::Size(9, 6);  // 方格标定板内角点数目（行，列）
	std::vector<std::vector<cv::Point2f>> imgLsPoints;
	std::vector<std::vector<cv::Point2f>> imgRsPoints;
	getImgsPoints(imgLs, imgLsPoints, stereoboard_size);  // 调用子函数
	getImgsPoints(imgRs, imgRsPoints, stereoboard_size);  // 调用子函数

	// 生成棋盘格每个内角点的空间三维坐标
	cv::Size stereosquareSize = cv::Size(10, 10);  //棋盘格每个方格的真实尺寸
	std::vector<std::vector<cv::Point3f>> stereoobjectPoints;
	for (int i = 0; i < imgLsPoints.size(); i++)
	{
		std::vector<cv::Point3f> tempPointSet;
		for (int j = 0; j < stereoboard_size.height; j++)
		{
			for (int k = 0; k < stereoboard_size.width; k++)
			{
				cv::Point3f realPoint;
				// 假设标定板为世界坐标系的z平面，即z=0
				realPoint.x = j * stereosquareSize.width;
				realPoint.y = k * stereosquareSize.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		stereoobjectPoints.push_back(tempPointSet);
	}

	//图像尺寸
	cv::Size stereoimageSize;
	stereoimageSize.width = imgLs[0].cols;
	stereoimageSize.height = imgLs[0].rows;

	cv::Mat stereoMatrix1, stereodist1, stereoMatrix2, stereodist2, stereorvecs, stereotvecs;
	cv::calibrateCamera(stereoobjectPoints, imgLsPoints, stereoimageSize, stereoMatrix1, stereodist1, stereorvecs, stereotvecs, 0);
	cv::calibrateCamera(stereoobjectPoints, imgRsPoints, stereoimageSize, stereoMatrix2, stereodist2, stereorvecs, stereotvecs, 0);
	
	//进行标定
	cv::Mat stereoR, stereoT, stereoE, stereoF;  //旋转矩阵、平移向量、本征矩阵、基本矩阵
	cv::stereoCalibrate(stereoobjectPoints, imgLsPoints, imgRsPoints, stereoMatrix1, stereodist1, stereoMatrix2, stereodist2, stereoimageSize, stereoR, stereoT, stereoE, stereoF, cv::CALIB_USE_INTRINSIC_GUESS);

	std::cout << "两个相机坐标系的旋转矩阵：" << std::endl << stereoR << std::endl;
	std::cout << "两个相机坐标系的平移向量：" << std::endl << stereoT << std::endl;

	//计算校正变换矩阵
	cv::Mat stereoR1, stereoR2, stereoP1, stereoP2, stereoQ;
	cv::stereoRectify(stereoMatrix1, stereodist1, stereoMatrix2, stereodist2, stereoimageSize, stereoR, stereoT, stereoR1, stereoR2, stereoP1, stereoP2, stereoQ, 0);

	//计算校正映射矩阵
	cv::Mat map11, map12, map21, map22;
	cv::initUndistortRectifyMap(stereoMatrix1, stereodist1, stereoR1, stereoP1, stereoimageSize, CV_16SC2, map11, map12);
	cv::initUndistortRectifyMap(stereoMatrix2, stereodist2, stereoR2, stereoP2, stereoimageSize, CV_16SC2, map21, map22);

	for (int i = 0; i < imgLs.size(); i++)
	{
		//进行校正映射
		cv::Mat img1r, img2r;
		cv::remap(imgLs[i], img1r, map11, map12, cv::INTER_LINEAR);
		cv::remap(imgRs[i], img2r, map21, map22, cv::INTER_LINEAR);

		//拼接图像
		cv::Mat result;
		cv::hconcat(img1r, img2r, result);

		//绘制直线，用于比较同一个内角点y轴是否一致
		cv::line(result, cv::Point(-1, imgLsPoints[i][0].y), cv::Point(result.cols, imgLsPoints[i][0].y), cv::Scalar(0, 0, 255), 2);
		cv::imshow("校正后结果", result);
		cv::waitKey(0);
	}
	cv::destroyAllWindows();

	return 0;
}