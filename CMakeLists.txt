cmake_minimum_required(VERSION 3.5)

project("install_test")

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 创建可执行文件
add_executable(${PROJECT_NAME} Src/install_test.cpp)

# 包含 OpenCV 头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 链接 OpenCV 库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
