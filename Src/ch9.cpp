#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

void orb_features(cv::Mat &gray, std::vector<cv::KeyPoint> &keypionts, cv::Mat &descriptions)
{
	cv::Ptr<cv::ORB> orb = cv::ORB::create(1000, 1.2f);
	orb->detect(gray, keypionts);
	orb->compute(gray, keypionts, descriptions);
}

void match_min(std::vector<cv::DMatch> matches, std::vector<cv::DMatch> & good_matches)
{
	double min_dist = 10000, max_dist = 0;
	for (int i = 0; i < matches.size(); i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	std::cout << "min_dist=" << min_dist << std::endl;
	std::cout << "max_dist=" << max_dist << std::endl;

	for (int i = 0; i < matches.size(); i++)
		if (matches[i].distance <= std::max(2 * min_dist, 20.0))
			good_matches.push_back(matches[i]);
}

//RANSAC算法实现
void ransac(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> queryKeyPoint, std::vector<cv::KeyPoint> trainKeyPoint, std::vector<cv::DMatch> &matches_ransac)
{
	//定义保存匹配点对坐标
	std::vector<cv::Point2f> srcPoints(matches.size()), dstPoints(matches.size());
	//保存从关键点中提取到的匹配点对的坐标
	for (int i = 0; i<matches.size(); i++)
	{
		srcPoints[i] = queryKeyPoint[matches[i].queryIdx].pt;
		dstPoints[i] = trainKeyPoint[matches[i].trainIdx].pt;
	}
	
	//匹配点对进行RANSAC过滤
	std::vector<int> inliersMask(srcPoints.size());
	//Mat homography;
	//homography = findHomography(srcPoints, dstPoints, RANSAC, 5, inliersMask);
	findHomography(srcPoints, dstPoints, cv::RANSAC, 5, inliersMask);
	//手动的保留RANSAC过滤后的匹配点对
	for (int i = 0; i<inliersMask.size(); i++)
		if (inliersMask[i])
			matches_ransac.push_back(matches[i]);
}

int main() {
	/* 角点检测 */
	/*
	绘制特征点函数
	cv::drawKeypoints(	InputArray image, 									// 输入图像
						const std::vector<KeyPoint>& keypoints, 			// 输入特征点
						InputOutputArray outImage, 							// 输出图像
						const Scalar& color = Scalar::all(-1), 				// 特征点的颜色
						DrawMatchesFlags flags = DrawMatchesFlags::DEFAULT 	// 绘制特征点的标志
						);
	
	cv::KeyPoint类
	class KeyPoint{
		float angle;  		// 关键点的角度
		int class_id;  		// 关键点的类别ID
		int octave;  		// 关键点所在的金字塔层数
		Point2f pt;  		// 关键点的坐标
		float response;  	// 关键点的响应值
		float size;  		// 关键点的直径
	};

	角点Harris值函数
	cv::cornerHarris(	InputArray src, 				// 输入图像
						OutputArray dst, 				// 输出图像
						int blockSize, 					// 角点检测中的邻域尺寸
						int ksize, 						// Sobel算子的半径尺寸,用于得到图像的梯度信息
						double k, 						// Harris角点检测系数R的权重参数
						int borderType = BORDER_DEFAULT // 像素外推法选择标志
						);

	Shi-Tomasi角点检测函数
	cv::goodFeaturesToTrack(InputArray image, 				// 输入图像
							OutputArray corners, 			// 输出角点
							int maxCorners, 				// 角点的最大数目
							double qualityLevel, 			// 角点的质量水平(角点阈值与最佳角点之间的关系)
							double minDistance, 			// 角点之间的最小欧式距离
							InputArray mask = noArray(), 	// 掩码图像
							int blockSize = 3, 				// 角点检测中的邻域尺寸(梯度协方差矩阵的尺寸)
							bool useHarrisDetector = false, // 是否使用Harris角点检测
							double k = 0.04 				// Harris角点检测郭崇中的常值权重系数
							);

	计算亚像素级别角点位置函数
	cv::cornerSubPix(	InputArray image, 			// 输入图像
						InputOutputArray corners, 	// 输入输出角点
						Size winSize, 				// 窗口尺寸()
						Size zeroZone, 				// 零区域尺寸(搜索区域中间“死区”的尺寸)
						TermCriteria criteria 		// 终止条件
						);
	*/
	cv::Mat lena = cv::imread("../Img/lena.jpg", cv::IMREAD_COLOR);
	if (!lena.data)
	{
		std::cout << "读取图像错误，请确认图像文件是否正确" << std::endl;
		return -1;
	}
	// 生产一个Lena图像的副本
	cv::Mat lenacopy1 = lena.clone();
	// 转成灰度图像
	cv::Mat lenagray1;
	cvtColor(lenacopy1, lenagray1, cv::COLOR_BGR2GRAY);

	// 计算Harris系数
	cv::Mat harris;
	int blockSize = 2;  // 邻域半径
	int apertureSize = 3;  // Sobel算子的半径尺寸
	cornerHarris(lenagray1, harris, blockSize, apertureSize, 0.04);
	
	// 归一化便于进行数值比较和结果显示
	cv::Mat harrisn;
	cv::normalize(harris, harrisn, 0, 255, cv::NORM_MINMAX);
	// 将图像的数据类型变成CV_8U
	cv::convertScaleAbs(harrisn, harrisn);
	
	// 寻找Harris角点
	std::vector<cv::KeyPoint> keyPoints1;
	for (int row = 0; row < harrisn.rows; row++)
	{
		for (int col = 0; col < harrisn.cols; col++)
		{
			int R = harrisn.at<uchar>(row, col);
			if (R > 125)// 阈值设定
			{
				// 向角点存入KeyPoint中
				cv::KeyPoint keyPoint;
				keyPoint.pt.y = row;
				keyPoint.pt.x = col;
				keyPoints1.push_back(keyPoint);
			}
		}
	}

	// 绘制角点与显示结果
	cv::drawKeypoints(lenacopy1, keyPoints1, lenacopy1);
	cv::imshow("系数矩阵", harrisn);
	cv::imshow("Harris角点", lenacopy1);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// 深拷贝用于第二种方法绘制角点
	cv::Mat lenacopy2 = lena.clone();
	cv::Mat lenagray2;
	cv::cvtColor(lenacopy2, lenagray2, cv::COLOR_BGR2GRAY);
	// Detector parameters

	// 提取角点
	int maxCorners = 100;  // 检测角点数目
	double quality_level = 0.01;  // 质量等级，或者说阈值与最佳角点的比例关系
	double minDistance = 0.04;  // 两个角点之间的最小欧式距离
	std::vector<cv::Point2f> corners2;
	goodFeaturesToTrack(lenagray2, corners2, maxCorners, quality_level, minDistance, cv::Mat(), 3, false);

	// 绘制角点
	std::vector<cv::KeyPoint> keyPoints2;  // 存放角点的KeyPoint类，用于后期绘制角点时用
	cv::RNG rng(10000);
	for (int i = 0; i < corners2.size(); i++) 
	{
		// 第一种方式绘制角点，用circle()函数绘制角点
		int b = rng.uniform(0, 256);
		int g = rng.uniform(0, 256);
		int r = rng.uniform(0, 256);
		circle(lenacopy2, corners2[i], 5, cv::Scalar(b, g, r), 2, 8, 0);

		// 将角点存放在KeyPoint类中
		cv::KeyPoint keyPoint;
		keyPoint.pt = corners2[i];
		keyPoints2.push_back(keyPoint);
	}

	// 第二种方式绘制角点，用drawKeypoints()函数
	drawKeypoints(lenacopy2, keyPoints2, lenacopy2);
	// 输出绘制角点的结果
	cv::imshow("用circle()函数绘制角点结果", lenacopy2);
	cv::imshow("通过绘制关键点函数绘制角点结果", lenacopy2);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// 彩色图像转成灰度图像
	cv::Mat lenagray3 = lenagray1.clone();

	// 提取角点
	int maxCorners3 = 100;  // 检测角点数目
	double quality_level3 = 0.01;  // 质量等级，或者说阈值与最佳角点的比例关系
	double minDistance3 = 0.04;  // 两个角点之间的最小欧式距离
	std::vector<cv::Point2f> corners3;
	goodFeaturesToTrack(lenagray3, corners3, maxCorners3, quality_level3, minDistance3, cv::Mat(), 3, false);
	
	// 计算亚像素级别角点坐标
	std::vector<cv::Point2f> cornersSub = corners3;  // 角点备份，方式别函数修改
	cv::Size winSize = cv::Size(5, 5);
	cv::Size zeroZone = cv::Size(-1, -1);	// 零区域尺寸,(-1,-1)表示没有死区
	cv::TermCriteria criteria = cv::TermCriteria(
		cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);
	cornerSubPix(lenagray3, cornersSub , winSize, zeroZone, criteria);

	// 输出初始坐标和精细坐标
	for (size_t i = 0; i < corners3.size(); i++)
	{
		std::string str = std::to_string(i);
		str = "第" + str + "个角点点初始坐标：";
		std::cout << str << corners3[i] << "   精细后坐标：" << cornersSub[i] << std::endl;
	}

	/* 特征点检测 */
	/*
	特征点检测函数
	cv::Feature2D::detect(	InputArray image, 				// 输入图像
							std::vector<KeyPoint>& keypoints, // 输出特征点
							InputArray mask = noArray() 		// 掩码图像
							);							// 返回值bool数据类型，表示是否检测到特征点的结果，true表示有，false表示无

	特征点描述子函数
	cv::Feature2D::compute(	InputArray image, 				// 输入图像
							std::vector<KeyPoint>& keypoints, // 输入特征点
							OutputArray descriptors 			// 输出特征点描述
							);							// 返回值bool数据类型，表示是否计算特征点描述的结果，true表示有，false表示无

	同时计算特征点关键点和描述子函数a
	cv::Feature2D::detectAndCompute(InputArray image, 				// 输入图像
									InputArray mask, 				// 掩码图像
									std::vector<KeyPoint>& keypoints, // 输出特征点
									OutputArray descriptors, 		// 输出特征点描述
									bool useProvidedKeypoints = false // 是否使用提供的特征点
									);						// 返回值bool数据类型，表示是否计算特征点关键点和描述子的结果，true表示有，false表示无
	*/

	// 生产一个Lena图像的副本
	cv::Mat lenacopy4 = lena.clone();

	// 创建SURF特征点类变量
	// 需要安装opencv_contrib库,进行Cmake时勾选OPENCV_ENABLE_NONFREE
	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(
		  400,  	// 关键点阈值
			4,  	// 检测关键点的金字塔组数
			3,  	// 检测关键点的金字塔层数
		false,  	// 是否使用扩展的描述子的标志,默认为false对应的是64维描述子，true对应的是128维描述子
		false		// 计算关键点方向,图像的局部梯度方向
	);

	// 计算SURF关键点
	std::vector<cv::KeyPoint> Keypoints4;
	detector->detect(lenacopy4, Keypoints4);  // 确定关键点 加载所有SURF指针到Keypoints4中

	// 计算SURF描述子
	cv::Mat descriptions4;
	detector->compute(lenacopy4, Keypoints4, descriptions4);  // 计算描述子

	// 绘制特征点
	cv::Mat imgAngel4 = lenacopy4.clone();
	// 绘制不含角度和大小的结果
	cv::drawKeypoints(lenacopy4, Keypoints4, lenacopy4, cv::Scalar(255,255,255));
	// 绘制含有角度和大小的结果
	cv::drawKeypoints(lenacopy4, Keypoints4, imgAngel4, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// 显示结果
	cv::imshow("SURF算法:不含角度和大小的结果", lenacopy4);
	cv::imshow("SURF算法:含有角度和大小的结果", imgAngel4);
	cv::waitKey(0); 
	cv::destroyAllWindows();
	
	// 生产一个Lena图像的副本
	cv::Mat lenacopy5 = lena.clone();
	// 创建 ORB 特征点类变量
	cv::Ptr<cv::ORB> orb = cv::ORB::create(	
		500, 					// 特征点数目
		1.2f, 					// 金字塔层级之间的缩放比例
		8, 						// 金字塔图像层数
		31, 					// 边缘阈值
		0, 						// 原图在金字塔中的层数
		2, 						// 生成描述子时需要用的像素点数目,默认为2,表示使用BRIEF描述子
		cv::ORB::HARRIS_SCORE, 	// 检测特征点的评价方法(例如使用Harris评价函数)
		31, 					// 生成描述子时关键点周围邻域的尺寸
		20 						// 计算 FAST 角点时像素值差值的阈值
	);

	//计算 ORB 关键点
	std::vector<cv::KeyPoint> Keypoints5;
	orb->detect(lenacopy5, Keypoints5); // 确定关键点

	//计算 ORB 描述子
	cv::Mat descriptions5;
	orb->compute(lenacopy5, Keypoints5, descriptions5); //计算描述子

	// 绘制特征点
	cv::Mat imgAngel5 = lenacopy5.clone();
	// 绘制不含角度和大小的结果
	cv::drawKeypoints(lenacopy5, Keypoints5, lenacopy5, cv::Scalar(255, 255, 255));
	// 绘制含有角度和大小的结果
	cv::drawKeypoints(lenacopy5, Keypoints5, imgAngel5, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// 显示结果

	cv::imshow("ORB算法:不含角度和大小的结果", lenacopy5);
	cv::imshow("ORB算法:含有角度和大小的结果", imgAngel5);
	cv::waitKey(0);
	cv::destroyAllWindows();

	/* 特征点匹配 */
	/*
	cv::DMatch类
	class DMatch{
		int distance; 	// 两个特征点描述子之间的距离
		int imgIdx; 	// 训练描述子来自的图像索引
		int queryIdx; 	// 查询图像中的特征点索引
		int trainIdx; 	// 训练图像中的特征点索引
	};

	特征点匹配函数
	cv::DescriptorMatcher::match(	InputArray queryDescriptors, 	// 查询描述子集合
									InputArray trainDescriptors, 	// 训练描述子集合
									OutputArray matches, 			// 两个集合描述子匹配结果
									InputArray mask = noArray() 	// 掩码图像
									);								
	
	距离最近描述子匹配函数
	cv::DesriptorMatcher::radiusMatch(	InputArray queryDescriptors, 				// 查询描述子集合
										InputArray trainDescriptors, 				// 训练描述子集合
										std::vector<std::vector<DMatch>>& matches, 	// 描述子匹配结果
										float maxDistance, 							// 最大距离(满足条件的距离阈值,描述子距离应小于该值)
										InputArray mask = noArray() 				// 掩码图像
										bool compactResult = false 				 	// 输出匹配结果数目是否与查询描述子数目相同的选择标志
										);											
	
	最近邻描述子匹配函数
	cv::DescriptorMatcher::knnMatch(	InputArray queryDescriptors, 				// 查询描述子集合
										InputArray trainDescriptors, 				// 训练描述子集合
										std::vector<std::vector<DMatch>>& matches, 	// 描述子匹配结果
										int k, 										// 每个查询描述子在训练描述子集合中的寻找的最优匹配结果的数目
										InputArray mask = noArray() 				// 掩码图像
										bool compactResult = false 					// 输出匹配结果数目是否与查询描述子数目相同的选择标志
										);											

	绘制特征点匹配结果
	cv::drawMatches(	InputArray img1, 											// 输入图像1
						const std::vector<KeyPoint>& keypoints1, 					// 输入特征点1
						InputArray img2, 											// 输入图像2
						const std::vector<KeyPoint>& keypoints2, 					// 输入特征点2
						const std::vector<DMatch>& matches1to2, 					// 输入特征点匹配结果
						OutputArray outImg, 										// 显示匹配结果的输出图像
						const Scalar& matchColor = Scalar::all(-1), 				// 连接线和关键点的颜色
						const Scalar& singlePointColor = Scalar::all(-1), 			// 没有匹配点的关键点的颜色
						const std::vector<char>& matchesMask = std::vector<char>(), // 匹配结果的掩码
						int flags = DrawMatchesFlags::DEFAULT 						// 绘制特征点匹配结果的标志
						);
	
	暴力匹配函数
	cv::BFMatcher::BFMatcher(	int normType = NORM_L2, 	// 描述子距离计算方法的标志
								bool crossCheck = false 	// 是否进行交叉验证的标志
								);							
	
	FLANN算法描述子匹配函数
	cv::FlannBasedMatcher::FlannBasedMatcher( 	const Ptr<flann::IndexParams> & indexParams = makePtr<flann::KDTreeIndexParams>(),	// 匹配时需要使用的搜索算法标志
												const Ptr<flann::SearchParams> & searchParams = makePtr<flann::SearchParams>()		// 递归遍历的次数，遍历次数越多越精确，但需要的时间越长
												);

	计算单应性矩阵函数
	cv::findHomography(	InputArray srcPoints, 				// 输入源图像的特征点的坐标
						InputArray dstPoints, 				// 输入目标图像的特征点的坐标
						int method, 						// 单应矩阵计算方法的标志
						double ransacReprojThreshold = 3, 	// RANSAC算法,重投影的最大误差
						OutputArray mask = noArray() 		// 输出掩码图像
						const int maxIters = 2000,			// RANSAC算法迭代的最大次数
                        const double confidence = 0.995		// 置信区间,取值范围为0~1
						);									// 返回值Mat数据类型，输出单应矩阵
	*/

	cv::Mat box, box_in_scene;  
	box = cv::imread("../Img/box.png");  
	box_in_scene = cv::imread("../Img/box_in_scene.png");

	if (!(box.data && box_in_scene.dataend))
	{
		std::cout << "读取图像错误，请确认图像文件是否正确" << std::endl;
		return -1;
	}

	// 提取ORB特征点
	std::vector<cv::KeyPoint> Keypoints6, Keypoints7;
	cv::Mat descriptions6, descriptions7;

	// 计算特征点
	orb_features(box, Keypoints6, descriptions6);
	orb_features(box_in_scene, Keypoints7, descriptions7);

	// 生成一个box图像的副本
	cv::Mat box_copy1 = box.clone();
	cv::Mat box_in_scene_copy1 = box_in_scene.clone();

	// 特征点暴力匹配
	std::vector<cv::DMatch> matches1;  // 定义存放匹配结果的变量
	cv::BFMatcher matcher1(cv::NORM_HAMMING);  // 定义特征点匹配的类，使用汉明距离
	matcher1.match(descriptions6, descriptions7, matches1);  // 进行特征点匹配
	std::cout << "matches=" << matches1.size() << std::endl;  // 匹配成功特征点数目
	
	// 通过汉明距离删选匹配结果
	double min_dist1 = 10000, max_dist1 = 0;
	for (int i = 0; i < matches1.size(); i++)
	{
		double dist = matches1[i].distance;
		if (dist < min_dist1) min_dist1 = dist;
		if (dist > max_dist1) max_dist1 = dist;
	}

	// 输出所有匹配结果中最大韩明距离和最小汉明距离
	std::cout << "min_dist1 =" << min_dist1 << std::endl;
	std::cout << "max_dist1 =" << max_dist1 << std::endl;

	// 将汉明距离较大的匹配点对删除
	std::vector<cv::DMatch>  good_matches1;
	for (int i = 0; i < matches1.size(); i++)
	{
		if (matches1[i].distance <= cv::max(2 * min_dist1, 20.0))
		{
			good_matches1.push_back(matches1[i]);
		}
	}
	std::cout << "good_min1 =" << good_matches1.size() << std::endl;  //剩余特征点数目

	// 绘制匹配结果
	cv::Mat outimg1, outimg2;
	cv::drawMatches(box, Keypoints6, box_in_scene, Keypoints7, matches1, outimg1);
	cv::drawMatches(box, Keypoints6, box_in_scene, Keypoints7, good_matches1, outimg2);
	cv::imshow("暴力匹配未筛选结果", outimg1);
	cv::imshow("暴力匹配最小汉明距离筛选", outimg2);

	cv::waitKey(0);
	cv::destroyAllWindows();  

	// 生成一个box图像的副本
	cv::Mat box_copy2 = box.clone();
	cv::Mat box_in_scene_copy2 = box_in_scene.clone();

	//判断描述子数据类型，如果数据类型不符需要进行类型转换，主要针对ORB特征点
	if ((descriptions6.type() != CV_32F) && (descriptions7.type() != CV_32F))
	{
		descriptions6.convertTo(descriptions6, CV_32F);
		descriptions7.convertTo(descriptions7, CV_32F);
	}

	// 特征点FLANN匹配
	std::vector<cv::DMatch> matches2;  // 定义存放匹配结果的变量
	cv::FlannBasedMatcher matcher2;  // 使用默认值即可
	matcher2.match(descriptions6, descriptions7, matches2); 	// 进行特征点匹配
	std::cout << "matches2 = " << matches2.size() << std::endl; 	// 匹配成功特征点数目

	//寻找距离最大值和最小值，如果是ORB特征点min_dist取值需要大一些
	double max_dist2 = 0; double min_dist2 = 100;
	for (int i = 0; i < descriptions6.rows; i++)
	{
		double dist = matches2[i].distance;
		if (dist < min_dist2) min_dist2 = dist;
		if (dist > max_dist2) max_dist2 = dist;
	}
	std::cout << " Max dist2 :" << max_dist2 << std::endl;
	std::cout << " Min dist2 :" << min_dist2 << std::endl;

	//将最大值距离的0.4倍作为最优匹配结果进行筛选
	std::vector<cv::DMatch> good_matches2;
	for (int i = 0; i < descriptions6.rows; i++)
	{
		if (matches2[i].distance < 0.40 * max_dist2)
		{
			good_matches2.push_back(matches2[i]);
		}
	}
	std::cout << "good_matches2 =" << good_matches2.size() << std::endl;  //匹配成功特征点数目

	//绘制匹配结果
	cv::Mat outimg3, outimg4;
	cv::drawMatches(box_copy2, Keypoints6, box_in_scene_copy2, Keypoints7, matches2, outimg3);
	cv::drawMatches(box_copy2, Keypoints6, box_in_scene_copy2, Keypoints7, good_matches2, outimg4);
	cv::imshow("FLANN匹配未筛选结果", outimg3);
	cv::imshow("FLANN匹配筛选结果", outimg4);

	cv::waitKey(0);
	cv::destroyAllWindows();

	// 生成一个box图像的副本
	cv::Mat box_copy3 = box.clone();
	cv::Mat box_in_scene_copy3 = box_in_scene.clone();

	//特征点匹配
	std::vector<cv::DMatch> matches3, good_min3, good_ransac3;
	cv::BFMatcher matcher3(cv::NORM_HAMMING);
	matcher3.match(descriptions6, descriptions7, matches3);
	std::cout << "matches3 =" << matches3.size() << std::endl;

	//最小汉明距离
	match_min(matches3, good_min3);
	std::cout << "good_min3 =" << good_min3.size() << std::endl;

	//用ransac算法筛选匹配结果
	ransac(good_min3, Keypoints6, Keypoints7, good_ransac3);
	std::cout << "good_matches.size3 =" << good_ransac3.size() << std::endl;

	//绘制匹配结果
	cv::Mat outimg5, outimg6, outimg7;
	cv::drawMatches(box_copy3, Keypoints6, box_in_scene_copy3, Keypoints7, matches3, outimg5);
	cv::drawMatches(box_copy3, Keypoints6, box_in_scene_copy3, Keypoints7, good_min3, outimg6);
	cv::drawMatches(box_copy3, Keypoints6, box_in_scene_copy3, Keypoints7, good_ransac3, outimg7);
	cv::imshow("未筛选结果", outimg5);
	cv::imshow("最小汉明距离筛选", outimg6);
	cv::imshow("ransac筛选", outimg7);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}