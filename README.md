# opencv4_test
---
## Chapter1
安装OpenCV
## Chapter2
图像存储容器Mat  
图像的读取与显示  
视频的读取与显示，调用摄像头  
数据保存XML和YMAL文件  
## Chapter3
图像像色空间RGB、HSV、YUV、Lab、灰度图  
像素统计、图像二值化、逻辑运算、LUT查找表像素归类  
图像连接、尺寸变换、翻转变换、仿射变换（旋转+平移）、透视变换、极坐标变换  
图像上绘制几何图像  
感兴趣区域  
图像金字塔（高斯金字塔、拉普拉斯金字塔）、上采样或下采样  
窗口交互操作  
## Chapter4 
图像直方图  
直方图归一化  
直方图比较  
直方图均衡化  
直方图匹配（手动的直方图均衡化）  
图像的模板匹配
## Chapter5
图像滤波  
图像卷积  
噪声：椒盐噪声、高斯噪声、泊松噪声、乘性噪声  
线性滤波：均值滤波、方框滤波、高斯滤波、可分离滤波  
非线性滤波：中值滤波、双边滤波  
图像边缘检测：Sobel算子、Scharr算子  
生成边缘检测滤波器：Laplacian算子、Canny算子
## Chapter6
图像连通域:
1. 欧式距离、街道距离、棋盘距离  
2. 4-领域、8-领域  
3. 常用的图像领域分析法：两遍扫描法、种子填充法  

腐蚀和膨胀  
开运算和闭运算  
形态学梯度  
顶帽运算  
黑帽运算  
## Chapter7
通过图像边缘检测算子得到二位边缘点集，再使用霍夫变换检测规定直线、圆形  
通过二维边缘点集，得到轮廓的面积或者周长等信息  
Hu矩具有旋转、平移和缩放不变性，利用此特点对比图像的Hu矩进行图像匹配  
定位并解码QR二维码  