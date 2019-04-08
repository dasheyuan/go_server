/*
 * Created on Thu Jan 26 2017 
 *
 * Copyright (c) 2017 Thomas Northall-Little
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "camera.h"
#include <iostream>
#include <chrono>
#include <thread>

Camera::Camera(int camID, float noise, bool flip, cv::Size photoSize) {
    cameraNumber = camID;
    cap = cv::VideoCapture(cameraNumber);
    CV_Assert(cap.isOpened());
    size = photoSize;
    cap.set(CV_CAP_PROP_FRAME_WIDTH, size.width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, size.height);
    updateCount = 0;
    frameNum = 0;
    numFrames = 0;
    noise_reduction_level = noise;
    shouldFlip = flip;
    threadActive = true;
    updateThread = std::thread(&Camera::update, this);
}

Camera::Camera(std::string videoName, float noise, bool flip, cv::Size photoSize) {
    isInputVideo = true;
    cap = cv::VideoCapture(videoName);
    CV_Assert(cap.isOpened());
    size = photoSize;
    int frameH = (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    int frameW = (int) cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int fps = (int) cap.get(CV_CAP_PROP_FPS);
    numFrames = (int) cap.get(CV_CAP_PROP_FRAME_COUNT);
    printf("vedio's \nwidth = %d\t height = %d\nvideo's fps = %d\t nums = %d\n", frameW, frameH, fps, numFrames);
    updateCount = 0;
    frameNum = 0;
    noise_reduction_level = noise;
    shouldFlip = flip;
    threadActive = true;
    updateThread = std::thread(&Camera::update, this);
}

Camera::~Camera() {
    threadActive = false;
    if (updateThread.joinable()) {
        updateThread.join();
    }
    std::cout << "Camera " << cameraNumber << " Halted: Number of Updates " << updateCount << "\n";
}

void Camera::update() {
    cv::Size emptySize(0, 0);
    while (threadActive) {
        cv::Mat in_frame;
        cap >> in_frame;

        if (size != emptySize) {
            cv::Mat resized_frame;
            cv::resize(in_frame, resized_frame, size, 0, 0, cv::INTER_CUBIC);
            resized_frame.copyTo(in_frame);
        }

        if (noise_reduction_level > 0) {
            cv::Mat bgr[3];
            cv::split(in_frame, bgr);

            std::vector <cv::Mat> array_of_Mats;

            cv::fastNlMeansDenoising(bgr[0], bgr[0], noise_reduction_level);
            cv::fastNlMeansDenoising(bgr[1], bgr[1], noise_reduction_level);
            cv::fastNlMeansDenoising(bgr[2], bgr[2], noise_reduction_level);

            array_of_Mats.push_back(bgr[0]);
            array_of_Mats.push_back(bgr[1]);
            array_of_Mats.push_back(bgr[2]);

            cv::Mat denoised_frame;
            cv::merge(array_of_Mats, denoised_frame);

            denoised_frame.copyTo(in_frame);
        }

        if (shouldFlip) {
            cv::Mat flipped;
            cv::flip(in_frame, flipped, 1);
            flipped.copyTo(in_frame);
        }

        in_frame.copyTo(frame);
        updateCount++;

        if (isInputVideo) {
            cv::imshow("Update Thread", frame);
            cv::waitKey(0);
        }

        if (isInputVideo && updateCount >= numFrames) {
            cap.set(CV_CAP_PROP_POS_FRAMES, 0);
            updateCount = 0;
        }
    }
}

cv::Mat Camera::getFrame() {
    while (frame.empty()); //blocking operation while the first frame comes through
    frameNum = updateCount;
    return frame;
}

cv::Mat Camera::getNewFrame() {
    while (frame.empty()); //blocking operation while the first frame comes through
    while (frameNum == updateCount); //blocking operation which makes the system wait for new frame
    frameNum = updateCount;
    return frame;
}

cv::Mat Camera::getAveragedFrame(int numFrames) {
    std::vector <cv::Mat> frames;
    frames[0] = getFrame();
    for (int i = 1; i < numFrames; i++) {
        frames[i] = getNewFrame();
    }
    return frames[0];
}

long int Camera::getFrameCount() {
    return updateCount;
}