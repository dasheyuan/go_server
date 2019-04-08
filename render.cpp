#include <opencv2/opencv.hpp>
#include <iomanip>


void matPrint(cv::Mat &img, int lineOffsY, cv::Scalar fontColor, const std::string &ss) {
    int fontFace = cv::FONT_HERSHEY_DUPLEX;
    double fontScale = 0.8;
    int fontThickness = 2;
    cv::Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    cv::Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;
    putText(img, ss, org, fontFace, fontScale, cv::Scalar(0, 0, 0), 5 * fontThickness / 2, 16);
    putText(img, ss, org, fontFace, fontScale, fontColor, fontThickness, 16);
}

void matPrint(cv::Mat &img, cv::Point point, cv::Scalar fontColor, const std::string &ss) {
    int fontFace = cv::FONT_HERSHEY_DUPLEX;
    double fontScale = 0.8;
    int fontThickness = 2;
    cv::Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    putText(img, ss, point, fontFace, fontScale, fontColor, fontThickness, 16);
}

void displayState(cv::Mat &canvas, bool bHelp, double fps, int state, long int frameCount) {
    cv::Scalar fontColorRed = cv::Scalar(0, 0, 255);
    cv::Scalar fontColorBule = cv::Scalar(255, 0, 0);
    cv::Scalar fontColorWhite = cv::Scalar(255, 255, 255);
    cv::Scalar fontColorNV = cv::Scalar(118, 185, 0);

    std::ostringstream ss;
    ss << "FPS = " << std::setprecision(1) << std::fixed << fps << " Frames = " << frameCount;
    matPrint(canvas, 0, fontColorWhite, ss.str());
    /*ss.str("");
    ss << "[" << canvas.cols << "x" << canvas.rows << "]";
    matPrint(canvas, 1, fontColorBule, ss.str());*/
    ss.str("");
    ss << "STATE = ";
    switch (state) {
        case 0:
            ss << "[INIT]";
            break;
        case 1:
            ss << "[SEARCH]";
            break;
        case 2:
            ss << "[TRACK]";
            break;
        case 3:
            break;
        default:
            break;
    }
    matPrint(canvas, 1, fontColorRed, ss.str());

    if (bHelp) {
        matPrint(canvas, 2, fontColorNV, "f - fullscreen/normal windows");
        matPrint(canvas, 3, fontColorNV, "r - switch to init state");
        matPrint(canvas, 4, fontColorNV, "s - switch to search state");
        matPrint(canvas, 5, fontColorNV, "q - quit");
    } else {
        matPrint(canvas, 2, fontColorNV, "h - toggle hotkeys help");
    }
}