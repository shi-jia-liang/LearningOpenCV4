#include <iostream>
#include <opencv2/opencv.hpp>

int main(){
	/* OpenCV与传统机器学习 */
	/*
	OpenCV中的传统机器学习算法
	1. K-近邻算法
	2. 支持向量机
	3. 决策树
	4. 随机森林
	5. Adaboost
	6. 贝叶斯分类器
	7. EM算法
	8. K均值聚类
	9. 层次聚类
	10. 主成分分析
	11. 独立成分分析
	12. 马尔科夫随机场
	13. 条件随机场
	14. 概率图模型
	15. 深度信念网络
	16. 限制玻尔兹曼机
	17. 自组织映射
	18. 线性判别分析
	19. 非线性判别分析
	20. 朴素贝叶斯
	21. 高斯混合模型
	22. 隐马尔科夫模型
	23. 最大熵模型
	24. 逻辑回归
	25. 神经网络
	26. 深度学习
	27. 卷积神经网络
	28. 循环神经网络
	29. 生成对抗网络
	30. 强化学习
	*/
	/*
	K均值聚类算法
	cv::kmeans(	InputArray data, 				// 输入数据
				int K, 							// 聚类数目
				InputOutputArray bestLabels, 	// 输出标签
				TermCriteria criteria, 			// 停止条件
				int attempts, 					// 尝试次数
				int flags, 						// 标志
				OutputArray centers 				// 输出中心
				);

	模型训练函数
	cv::StatModel::train(	InputArray samples, 		// 输入样本
							int layout, 				// 布局
							InputArray responses 		// 输出响应
							);
	模型保存函数
	cv::TrainData::create(	InputArray samples, 		// 输入样本
							int layout, 				// 布局
							InputArray responses, 		// 输出响应
							InputArray varIdx = noArray(), // 变量索引
							InputArray sampleIdx = noArray(), // 样本索引
							InputArray sampleWeights = noArray(), // 样本权重
							InputArray varType = noArray() // 变量类型
							);

	模型预测函数(利用模型对新数据进行预测)
	cv::StatModel::predict(	InputArray samples, 		// 输入样本
							OutputArray results, 		// 输出结果
							int flags = 0 				// 标志
							);							// 返回值bool数据类型，表示是否预测的结果，true表示有，false表示无
	
	模型加载函数
	cv::Algorithm::load(const String& filename, 	// 文件名
						const String& objname = String() // 对象名
						);
		
	K近邻模型对象创建函数
	cv::KNearst::create(int defaultK = 32, 		// 默认K值
						int isRegression = false, 	// 是否回归
						int maxK = INT_MAX 			// 最大K值
						);
	
	K近邻模型训练函数
	cv::KNearest::findNearest(	InputArray samples, 		// 输入样本
								int k, 					// K值
								OutputArray results, 		// 输出结果
								OutputArray neighborResponses, // 输出邻居响应
								OutputArray dists 			// 输出距离
								);							// 返回值float数据类型，表示距离的结果
	
	cv::DTrees::create(	);  // 创建决策树对象
	cv::RTrees::create(	);  // 创建随机森林对象
	cv::SVM::create(	);  // 创建SVM对象
	*/
	
	/* OpenCV与深度神经网络应用实例 */
	/*
	OpenCV中的深度学习应用实例
	1. 图像分类
	2. 目标检测
	3. 人脸识别
	*/
	/*
	加载已有深度神经网络模型
	cv::dnn::readNet(	const String& model, 		// 模型文件
						const String& config = "", 	// 配置文件
						const String& framework = "" // 框架
						);							// 返回值cv::dnn::Net数据类型，表示神经网络模型
	
	向深度神经网络模型中输入数据
	cv::dnn::Net::setInput(	InputArray blob, 		// 输入数据
							const String& name = "" 	// 名称
							);						// 返回值void数据类型，表示无返回值

	转换输入到深度神经网络模型中的图像尺寸
	cv::dnn::blobFromImages(	InputArray images, 		// 输入图像
								double scalefactor = 1.0, // 缩放因子
								const Size& size = Size(), // 尺寸
								const Scalar& mean = Scalar(), // 均值
								bool swapRB = false, 		// 交换RB通道
								bool crop = false 			// 裁剪
								);							// 返回值cv::Mat数据类型，表示图像数据
	*/
	return 0;
}