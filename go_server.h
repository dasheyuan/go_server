#ifndef GO_SERVER_H
#define GO_SERVER_H

#include <iostream>
#include <iomanip>
#include <utility>
#include <thread>
#include <chrono>
#include <functional>
#include <atomic>
#include <memory> // share_ptr

//OPENPOSE Headers
#include <openpose/headers.hpp>
#include <twaipose/pose/poseWrapper.hpp>
#include <twaipose/utilities/headers.hpp>
//GOTURN Headers
#include <tracker/tracker.h>
#include <helper/bounding_box.h>
//TCP Socket Headers
#include "practicalsockets/practicalsocket.h"
//Camera Headers
#include "multithread_camera/camera.h"
//INI Config File Parser Headers-Only
#include "ini_parser.hpp"

enum class BackEndState : unsigned char {
    INIT = 0,
    SEARCH = 1,
    TRACK = 2,
    Size,
};


//cv::Mat TCP transfer format 
#define PACKAGE_NUM 32
#define IMG_WIDTH 1280
#define IMG_HEIGHT 720

#define BUFFER_SIZE IMG_WIDTH*IMG_HEIGHT*3/PACKAGE_NUM
struct sentbuf {
    char buf[BUFFER_SIZE];
    int flag;
};


//search timeout (now-start) > 24
//std::chrono::time_point<std::chrono::high_resolution_clock> start;
//std::chrono::time_point<std::chrono::high_resolution_clock> now;


//std::chrono::time_point<std::chrono::high_resolution_clock> fpsstart;
//std::chrono::time_point<std::chrono::high_resolution_clock> fpsnow;


//state define
bool track_init_state = true;
bool target_search_state = false;
bool tracking_state = false;

TCPSocket *poseclntSock = nullptr;
const int BUFSIZE = 1024;
std::string json;

bool help_screen = false;
bool full_screen = false;
cv::Mat targetHeadCrop(227, 227, CV_8UC3, cv::Scalar(0, 0, 0));
std::array<std::pair<double, cv::Mat>, 1> targetHeadCrops;

//thread callback
void ImageServer(Camera *c);

void PoseServer();

void CommandServer();


extern void matPrint(cv::Mat &img, int lineOffsY, cv::Scalar fontColor, const string &ss);

extern void matPrint(cv::Mat &img, cv::Point point, cv::Scalar fontColor, const string &ss);

extern void displayState(cv::Mat &canvas, bool bHelp, double fps, int state, long int frameCount);


bool isTargetCoveredwithFace(op::Array<float> &keypoints, std::vector <op::Rectangle<float>> &faceRectangles,
                             op::Rectangle<float> bbox, int target);

double
pickMaxFaceSimilarity(cv::Mat &curr_frame, std::vector <op::Rectangle<float>> &faceRectangles, cv::Mat &targetHeadCrop,
                      int &target, RegressorBase *regressor);

#endif