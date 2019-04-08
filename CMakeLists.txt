cmake_minimum_required(VERSION 2.8.12)
project(go_server)

## Compile as C++11, supported in ROS Kinetic and newer
#add_compile_options(-std=c++11)
add_definitions(-std=c++11)


find_package(CUDA REQUIRED)
# Note: If can't find CUDA, please uncomment the below line and set the path manually
# set(CUDA_INCLUDE_DIRS /path_to_cuda/include)
include_directories(${CUDA_INCLUDE_DIRS})
message("CUDA_INCLUDE_DIRS is ${CUDA_INCLUDE_DIRS}")


find_package(OpenCV REQUIRED)
message("Open CV version is ${OpenCV_VERSION}")
# Uncomment for no display (e.g. AWS instance)
# set(Opencv_DEFINITIONS -DNO_DISPLAY)
add_definitions(${Opencv_DEFINITIONS})
message("Opencv_DEFINITIONS is ${Opencv_DEFINITIONS}")

find_package(Boost COMPONENTS system filesystem regex REQUIRED)


set(OPENPOSE_DIR "$ENV{OPENPOSE_ROOT}")

if (OPENPOSE_DIR)               # <--- Use 'BOOST_DIR', not 'DEFINED ${BOOST_DIR}'
    message(STATUS "OPENPOSE_HOME at ${OPENPOSE_DIR}")
else ()
    message(FATAL_ERROR "Undefined OPENPOSE_HOME env var.")
endif ()

set(GOTURN_DIR "~/CLionProjects/GOTURN_3")

# include_directories(include)
include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${OPENPOSE_DIR}/include
        ${OPENPOSE_DIR}/3rdparty/caffe/include
        ${OPENPOSE_DIR}/3rdparty/caffe/.build_release/src
        ${GOTURN_DIR}/src
        ${GOTURN_DIR}/src/native
)
# openpose node
add_executable(go_server
        main.cpp
        multithread_camera/camera.h
        multithread_camera/camera.cpp
        practicalsockets/practicalsocket.h
        practicalsockets/practicalsocket.cpp
        render.cpp
        )
target_link_libraries(go_server
        ${OPENPOSE_DIR}/build/lib/libopenpose.so
        ${OPENPOSE_DIR}/build/lib/libtwaipose.so
        ${OPENPOSE_DIR}/3rdparty/caffe/.build_release/lib/libcaffe.so
        ${GOTURN_DIR}/build/libGOTURN.so
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        )
