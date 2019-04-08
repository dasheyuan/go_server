#include "go_server.h"

auto start = std::chrono::high_resolution_clock::now();

int main(int argc, char *argv[]) {
    op::log("WallCross Game Server Start.");
    op::log("Reading Config File");
    INI::Parser p("./config.ini");

    const std::string model_file = p.top()("goturn")["model_file"];
    const std::string trained_file = p.top()("goturn")["trained_file"];
    std::string target_jpg = p.top()("goturn")["target_jpg"];
    std::string resolution = p.top()("openpose")["resolution"];
    std::string model_folder = p.top()("openpose")["model_folder"];
    std::string rank_jpg = p.top()("game")["rank_jpg"];
    std::string video = p.top()("default")["video"];
    std::string camera_id = p.top()("default")["camera_id"];

    op::log("GOTURN:\n--model file:" + model_file + " \n--trained file:" + trained_file);
    op::log("OpenPose:\n--model file:" + model_folder);
    op::log("Game:\n--image save path:" + rank_jpg);

    op::log("Init Video Capture:");
    Camera cam(video, 0, false, cv::Size(1280, 720));

    op::log("Init threads:");
    std::thread image_server_thread(ImageServer, &cam);
    std::thread pose_server_thread(PoseServer);
    std::thread command_server_thread(CommandServer);

    pose_server_thread.detach();
    image_server_thread.detach();
    command_server_thread.detach();

    op::log("Instantiate openpose.");
    op::HandDetector hand_detector(op::PoseModel::COCO_18);
    op::FaceDetector face_detector(op::PoseModel::COCO_18);
    tp::PoseWrapper pose_wrapper(model_folder, resolution);
    tp::KeypointJsonPrinter keypointJsonPrinter;

    op::log("Instantiate goturn.");
    Regressor *regressor = new Regressor(model_file, trained_file, 0, false);
    Tracker *tracker = new Tracker(false);


    BoundingBox bbox_estimated;
    BoundingBox target_face_box;
    op::Rectangle<float> target_hand_rectangle;
    op::Rectangle<float> target_face_rectangle;
    op::Rectangle<float> target_estimated_face_rectangle;

    cv::Mat frame_in, frame_out_to_display;
    int target_lost_frame = 0;
    int target_covered_frame = 0;

    cv::namedWindow("result", CV_WINDOW_NORMAL);
    cv::resizeWindow("result", 1280, 720);

    BackEndState currentState = BackEndState::INIT;


    while (true) {
        const auto fpsstart = std::chrono::high_resolution_clock::now();
        frame_in = cam.getFrame();

        if (currentState == BackEndState::INIT) {
            double scaleInputToOutput;
            auto poseKeypoints = pose_wrapper.getPoseKeypoint(frame_in, frame_out_to_display, scaleInputToOutput);
            auto numberPeople = 0;
            numberPeople = op::fastMax(numberPeople, poseKeypoints.getSize(0));
            const auto numberBodyParts = poseKeypoints.getSize(1);
            const auto handRectangles = hand_detector.detectHands(poseKeypoints, scaleInputToOutput);
            const auto faceRectangles = face_detector.detectFaces(poseKeypoints, scaleInputToOutput);
            const auto threshold = 0.8f;
            int personIndex = 0;
            bool find = false;

            for (auto person = 0; person < numberPeople; person++) {

                auto averageScore = op::getAverageScore(poseKeypoints, person);
                if (averageScore < 0.6) continue;


                auto area = op::getKeypointsArea(poseKeypoints, person, numberBodyParts, threshold);
                if (area < 30000) continue;

                const auto RShoulderOffset = (person * numberBodyParts + 2);
                const auto RElbowOffset = (person * numberBodyParts + 3);
                const auto RWristOffset = (person * numberBodyParts + 4);

                const auto *posePtr = &poseKeypoints.at(person * poseKeypoints.getSize(1) * poseKeypoints.getSize(2));

                const auto wristScoreAbove = (posePtr[RWristOffset * 3 + 2] > threshold);
                const auto elbowScoreAbove = (posePtr[RElbowOffset * 3 + 2] > threshold);
                const auto shoulderScoreAbove = (posePtr[RShoulderOffset * 3 + 2] > threshold);


                if (wristScoreAbove && elbowScoreAbove && shoulderScoreAbove) {
                    const auto wristY = posePtr[RWristOffset * 3 + 1];
                    const auto elbowY = posePtr[RElbowOffset * 3 + 1];
                    const auto shoulderY = posePtr[RShoulderOffset * 3 + 1];

                    if (wristY <= elbowY && wristY <= shoulderY && wristY > 0) {
                        const auto elbowX = posePtr[RElbowOffset * 3];
                        const auto shoulderX = posePtr[RShoulderOffset * 3];

                        auto ratioElbowShoulder = fabsf(elbowY - shoulderY) /
                                                  sqrtf(fabsf(elbowX - shoulderX) * fabsf(elbowX - shoulderX) +
                                                        fabsf(elbowY - shoulderY) * fabsf(elbowY - shoulderY));

                        if (ratioElbowShoulder <= 0.5) {
                            find = true;
                            personIndex = person;
                            break;
                        }
                    }
                }
            }


            if (find) {
                target_hand_rectangle.x = handRectangles.at(personIndex).at(1).x;
                target_hand_rectangle.y = handRectangles.at(personIndex).at(1).y;
                target_hand_rectangle.width = handRectangles.at(personIndex).at(1).width;
                target_hand_rectangle.height = handRectangles.at(personIndex).at(1).height;

                cv::rectangle(frame_out_to_display, tp::rectangleToCvRectangle(target_hand_rectangle),
                              cv::Scalar(255, 255, 0), 3);

                const auto faceRectangle = faceRectangles.at(personIndex);
                if (faceRectangle.width > 0 && faceRectangle.height > 0 && faceRectangle.x > 0 && faceRectangle.y > 0) {
                    /*target_face_box.x1_ = faceRectangle.x + faceRectangle.width / 5;
                    target_face_box.y1_ = faceRectangle.y;
                    target_face_box.x2_ = faceRectangle.x + faceRectangle.width / 5 * 4;
                    target_face_box.y2_ = faceRectangle.y + faceRectangle.height / 5 * 4;*/
                    target_face_box.x1_ = faceRectangle.x;
                    target_face_box.y1_ = faceRectangle.y;
                    target_face_box.x2_ = faceRectangle.x + faceRectangle.width;
                    target_face_box.y2_ = faceRectangle.y + faceRectangle.height;


                    cv::Mat ROI(frame_in, cv::Rect(faceRectangle.x, faceRectangle.y * (0.95), faceRectangle.width,
                                                   faceRectangle.height));
                    cv::resize(ROI, targetHeadCrop, cv::Size(227, 227), (0, 0), (0, 0), cv::INTER_LINEAR);
                    //Copy the data into new matrix
                    //ROI.copyTo(croppedImage);
                    imwrite(rank_jpg, targetHeadCrop);


                    cv::Mat ROI2(frame_in, cv::Rect(target_face_box.x1_, target_face_box.y1_,
                                                    target_face_box.x2_ - target_face_box.x1_,
                                                    target_face_box.y2_ - target_face_box.y1_));
                    /*cv::imshow("ROI2",ROI2);
                    cv::waitKey(0);*/


                    cv::resize(ROI2, targetHeadCrop, cv::Size(227, 227), (0, 0), (0, 0), cv::INTER_LINEAR);
                    targetHeadCrops[0] = std::make_pair(1.0, targetHeadCrop);
                    //targetHeadCrops[1] = std::make_pair(1.0, targetHeadCrop);
                    /*cv::imshow("croppedImage", targetHeadCrop);
                    cv::imshow("a1", targetHeadCrops[0].second);
                    cv::imshow("a2", targetHeadCrops[1].second);*/
                    imwrite(target_jpg, targetHeadCrop);

#ifndef NO_DEBUG
                    printf("OpenposeNode##Openpose --> Goturn face\n");
                    printf("%f,%f,%f,%f\n", target_face_box.x1_, target_face_box.y1_, target_face_box.x2_,
                           target_face_box.y2_);
#endif
                    tracker->Init(frame_in, target_face_box, regressor);

                    currentState = BackEndState::TRACK;
                }
            }//end if (find)
        }//end if (track_init_state)

        if (currentState == BackEndState::SEARCH) {
            BoundingBox search_result;
            double similarity = 0.f;
            double scaleInputToOutput;
            auto poseKeypoints = pose_wrapper.getPoseKeypoint(frame_in, frame_out_to_display, scaleInputToOutput);
            std::vector <op::Rectangle<float>> faces = face_detector.detectFaces(poseKeypoints, scaleInputToOutput);
#ifndef NO_DEBUG
            for (size_t i = 0; i < faces.size(); i++) {
                //auto personRectangle = op::getKeypointsRectangle(poseKeypoints, i, 18,0.1f);
                //cv::rectangle(frameDisp, tp::rectangleToCvRectangle(personRectangle), cv::Scalar(0, 0, 255), 3);//Red
                cv::rectangle(frame_out_to_display, tp::rectangleToCvRectangle(faces.at(i)), cv::Scalar(0, 0, 255),
                              3);//Red
            }
#endif
            for (size_t i = 0; i < faces.size(); i++) {

                if (faces.at(i).x <= 0 || faces.at(i).y <= 0 || (faces.at(i).x + faces.at(i).width) >= 1280 ||
                    (faces.at(i).y + faces.at(i).height) >= 720 || faces.at(i).width < 1 || faces.at(i).height < 1) {
                    cv::imshow("result", frame_out_to_display);
                    continue;
                };

                if (faces.at(i).area() < 9000) {
                    //printf("target is small...");
                    continue;
                }

                cv::Rect search_target;
                /*search_target.x = faces.at(i).x + faces.at(i).width / 4;
                search_target.y = faces.at(i).y;
                search_target.width =  faces.at(i).width / 4 * 2;
                search_target.height = faces.at(i).height / 4 * 3;*/
                search_target.x = faces.at(i).x;
                search_target.y = faces.at(i).y;
                search_target.width = faces.at(i).width;
                search_target.height = faces.at(i).height;

                cv::Mat roi_curr(frame_in, search_target);
                cv::resize(roi_curr, roi_curr, cv::Size(227, 227), (0, 0), (0, 0), cv::INTER_LINEAR);
                std::ostringstream sss;
                sss << "./a" << i << ".jpg";
                std::string filename = sss.str();
                cv::imwrite(filename, roi_curr);
                cv::imwrite("./b1.jpg", targetHeadCrops[0].second);
                regressor->Regress(frame_in, roi_curr, targetHeadCrops[0].second, &search_result);
                similarity = search_result.p_;
                //regressor->Regress(frame_in, roi_curr, targetHeadCrops[1].second, &search_result);
                //similarity2 = search_result.p_;
                //std::cout << "Similarity: " << similarity << "\t\t" << "Similarity2:" << similarity2 << "\n";
                std::cout << "Similarity: " << similarity << "\n";
                //similarity = similarity > similarity2 ? similarity : similarity2;

                ostringstream ss;
                ss << "P:" << setprecision(2) << similarity;
                cv::Scalar fontColorNV = cv::Scalar(0, 255, 0);
                matPrint(frame_out_to_display, cv::Point(search_target.x, search_target.y), fontColorNV, ss.str());
                if (similarity > 0.68) {
                    if (isTargetCoveredwithFace(poseKeypoints, faces, faces.at(i), i)) {
                        std::cout << "target is covered while searching...\n";
                        continue;
                    } else if (similarity > 0.8) {
                        target_face_box.x1_ = search_target.x;
                        target_face_box.y1_ = search_target.y;
                        target_face_box.x2_ = search_target.x + search_target.width;
                        target_face_box.y2_ = search_target.y + search_target.height;
                        cv::rectangle(frame_out_to_display, tp::rectangleToCvRectangle(faces.at(i)),
                                      cv::Scalar(255, 255, 255), 3);
                        tracker->Init(frame_in, target_face_box, regressor);

                        currentState = BackEndState::TRACK;
                    }
                }
            }//end of for (size_t i = 0; i < faces.size(); i++)

            const auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= 30) {
                //std::cout << "search timeout : " << 24 << std::endl;
                //currentState = BackEndState::INIT;
            }
        }

        if (currentState == BackEndState::TRACK) {
            tracker->Track(frame_in, regressor, &bbox_estimated);

            target_estimated_face_rectangle.x = bbox_estimated.x1_;
            target_estimated_face_rectangle.y = bbox_estimated.y1_;
            target_estimated_face_rectangle.width = bbox_estimated.x2_ - bbox_estimated.x1_;
            target_estimated_face_rectangle.height = bbox_estimated.y2_ - bbox_estimated.y1_;

            int person = -1, person2 = -1;
            double scaleInputToOutput;
            auto poseKeypoints = pose_wrapper.getPoseKeypoint(frame_in, frame_out_to_display, scaleInputToOutput);
            auto faceRectangles = face_detector.detectFaces(poseKeypoints, scaleInputToOutput);
            const auto thresholdRectangle = 0.1f;
            const auto numberKeypoints = poseKeypoints.getSize(1);
            const auto faceIOU = tp::pickMaxbboxOverlapRation(faceRectangles, target_estimated_face_rectangle, person);

            const auto faceSimilarity = pickMaxFaceSimilarity(frame_in, faceRectangles, targetHeadCrop, person2,
                                                              regressor);


#ifndef NO_DEBUG
            //std::cout << "person1:" << person <<" "<< "person2:" << person2 <<" similarity:"<< faceSimilarity <<"\n";

            for (size_t i = 0; i < faceRectangles.size(); i++) {
                //auto personRectangle = op::getKeypointsRectangle(poseKeypoints, i, numberKeypoints,
                //	thresholdRectangle);
                //cv::rectangle(frameDisp, tp::rectangleToCvRectangle(personRectangle), cv::Scalar(0, 0, 255), 3);//Red
                cv::rectangle(frame_out_to_display, tp::rectangleToCvRectangle(faceRectangles.at(i)),
                              cv::Scalar(0, 0, 255), 3);//Red
            }
#endif

            if (person != -1 && faceIOU > 0.32) {
                target_face_rectangle = faceRectangles.at(person);


                if (target_face_rectangle.x <= 0 || target_face_rectangle.y <= 0 ||
                    (target_face_rectangle.x + target_face_rectangle.width) >= 1280 ||
                    (target_face_rectangle.y + target_face_rectangle.height) >= 720) {
                    cv::imshow("result", frame_out_to_display);
                    continue;
                };

                bbox_estimated.x1_ = target_face_rectangle.x + target_face_rectangle.width / 4;
                bbox_estimated.y1_ = target_face_rectangle.y;
                bbox_estimated.x2_ = target_face_rectangle.x + target_face_rectangle.width / 4 * 3;
                bbox_estimated.y2_ = target_face_rectangle.y + target_face_rectangle.height / 4 * 3;

                BoundingBox search_result;
                double similarity = 0.f;

                cv::Mat roi_curr(frame_in, cv::Rect(bbox_estimated.x1_, bbox_estimated.y1_,
                                                    bbox_estimated.x2_ - bbox_estimated.x1_,
                                                    bbox_estimated.y2_ - bbox_estimated.y1_));
                cv::resize(roi_curr, roi_curr, cv::Size(227, 227), (0, 0), (0, 0), cv::INTER_LINEAR);
                regressor->Regress(frame_in, roi_curr, targetHeadCrop, &search_result);
                similarity = search_result.p_;
                /*if(similarity < 0.45)
                    targetHeadCrops[0] = std::make_pair(similarity, roi_curr);*/

                /*cv::imshow("a1", targetHeadCrops[0].second);
                cv::imshow("a2", targetHeadCrops[1].second);*/

                ostringstream ss;
                ss << similarity;
                cv::Scalar fontColorNV = cv::Scalar(200, 0, 0);
                //matPrint(frameDisp, cv::Point(target_face_rectangle.x, target_face_rectangle.y), fontColorNV, ss.str());

                tracker->Init(frame_in, bbox_estimated, regressor);


                if (similarity < 0.52) {

                    if (isTargetCoveredwithFace(poseKeypoints, faceRectangles, target_face_rectangle, person)) {
                        std::cout << "Occlusion!--Similarity: " << similarity << "faceIOU:" << faceIOU << "\n";
                        target_covered_frame++;
                        target_face_rectangle.x = 0;
                        target_face_rectangle.y = 0;
                        target_face_rectangle.width = 0;
                        target_face_rectangle.height = 0;

                        if (target_covered_frame > 0) {
#ifndef NO_DEBUG
                            printf("target_covered_frame=%d\n", target_covered_frame);
#endif
                            target_covered_frame = 0;


                            currentState = BackEndState::SEARCH;
                            const auto start = std::chrono::high_resolution_clock::now();
                        }
                        continue;
                    }
                }


                op::Array<float> targetKeypoints;
                if (poseKeypoints.getSize(0) > 0)
                    targetKeypoints.reset({1, 18, 3});
                else
                    targetKeypoints.reset();

                auto bodyPart = 0;
                for (; bodyPart < 18; bodyPart++) {
                    const auto baseOffset = (person * 18 + bodyPart) * 3;
                    targetKeypoints[bodyPart * 3] = poseKeypoints[baseOffset];
                    targetKeypoints[bodyPart * 3 + 1] = poseKeypoints[baseOffset + 1];
                    targetKeypoints[bodyPart * 3 + 2] = poseKeypoints[baseOffset + 2];
                }
                const std::vector <std::pair<op::Array < float>, std::string>>
                keypointVector{
                        std::make_pair(targetKeypoints, "pose_keypoints")};
                bool humanReadable = true;
                bool stringFloated = true;
                keypointJsonPrinter.print(keypointVector, humanReadable, json);
                try {
                    if (poseclntSock != nullptr)
                        poseclntSock->send(json.c_str(), json.size());
                }
                catch (SocketException &e) {
                    std::cerr << "Pose Handle Client : Send Failed...client may be exit." << std::endl;
                    delete poseclntSock;
                    poseclntSock = nullptr;
                }
                //std::cout << json << std::endl;
#ifndef NO_DEBUG
                cv::rectangle(frame_out_to_display, tp::rectangleToCvRectangle(target_estimated_face_rectangle),
                              cv::Scalar(255, 0, 0), 3);//Bule
#endif
            }// end of if (person != -1 && faceIOU > 0.3)
            else {
                target_lost_frame++;

                target_face_rectangle.x = 0;
                target_face_rectangle.y = 0;
                target_face_rectangle.width = 0;
                target_face_rectangle.height = 0;

                if (target_lost_frame > 20) {
#ifndef NO_DEBUG
                    printf("Target lost...\n");
#endif
                    target_lost_frame = 0;


                    currentState = BackEndState::SEARCH;
                    start = std::chrono::high_resolution_clock::now();
                }
            }

        }// end of if (currentState == BackEndState::TRACK)

        const auto fpsnow = std::chrono::high_resolution_clock::now();
        auto fps = 1000 / std::chrono::duration_cast<std::chrono::milliseconds>(fpsnow - fpsstart).count();
        displayState(frame_out_to_display, help_screen, fps, (int) currentState, cam.getFrameCount());
        cv::imshow("result", frame_out_to_display);

        char key = (char) cv::waitKey(5);
        if (key == 27) {
            break;
        }
        switch (key) {
            case ' ':
                //useGPU = !useGPU;
                break;
            case 'f':
                full_screen = !full_screen;
                if (full_screen)
                    cv::setWindowProperty("result", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
                else
                    cv::setWindowProperty("result", CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
                break;
            case 'q':
                delete regressor;
                delete tracker;
                return 0;
                break;
            case 'r':
                currentState = BackEndState::INIT;
                break;
            case 's':
                currentState = BackEndState::SEARCH;

                start = std::chrono::high_resolution_clock::now();
                break;
            case 'h':
                help_screen = !help_screen;
                break;
            default:
                break;
        }//end of switch(key)
    } //end of while

    return 0;
}

void ImageHandleClient(TCPSocket *sock, Camera *c) {
    struct sentbuf data;

    cout << "Image Client " + sock->getForeignAddress() << ":" << sock->getForeignPort() << " connected..." << endl;


    for (;;) {
        std::this_thread::sleep_for(std::chrono::microseconds(300));

        cv::Mat frame = c->getFrame();

        for (int k = 0; k < PACKAGE_NUM; k++) {
            int num1 = IMG_HEIGHT / PACKAGE_NUM * k;
            for (int i = 0; i < IMG_HEIGHT / PACKAGE_NUM; i++) {
                int num2 = i * IMG_WIDTH * 3;
                uchar *ucdata = frame.ptr<uchar>(i + num1);
                for (int j = 0; j < IMG_WIDTH * 3; j++) {
                    data.buf[num2 + j] = ucdata[j];
                }
            }

            if (k == PACKAGE_NUM - 1)
                data.flag = 2;
            else
                data.flag = 1;

            try {
                sock->send((char *) (&data), sizeof(data));
            }
            catch (SocketException &e) {
                cerr << "ImageHandleClient : Send Failed...client may be exit." << endl;
                delete sock;
                return;
            }
        }
    }
}

void ImageServer(Camera *c) {
    op::log("Image Server Thread Start...Port:9999");
    unsigned short imageServPort = 9999;    // First arg:  local port

    try {
        TCPServerSocket servSock(imageServPort);   // Socket descriptor for server
        for (;;) {      // Run forever
            // Create separate memory for client argument
            TCPSocket *clntSock = servSock.accept();

            std::thread ImageHandleClientThread(ImageHandleClient, clntSock, c);
            ImageHandleClientThread.detach();
        }
    }
    catch (SocketException &e) {
        cerr << e.what() << endl;
        exit(1);
    }
}


void poseHandleClient(TCPSocket *sock) {
    cout << "Pose Client " << sock->getForeignAddress() << ":" << sock->getForeignPort() << " connected..." << endl;
    try {
        for (;;) {

            if (json.length() == 0) continue;
            sock->send(json.c_str(), json.size());
        }
    }
    catch (SocketException &e) {
        cerr << "poseHandleClient : Send Failed...client may be exit." << endl;
        delete sock;
        return;
    }
}


void PoseServer() {
    op::log("Pose Server Thread Start...Port:9998");
    unsigned short poseServPort = 9998;    // First arg:  local port

    try {
        TCPServerSocket poseServSock(poseServPort);   // Socket descriptor for server
        for (;;) {      // Run forever
            // Create separate memory for client argument
            /*if (poseclntSock != nullptr)
            {
            delete poseclntSock;
            poseclntSock == nullptr;
            }*/
            poseclntSock = poseServSock.accept();
            cout << "Pose Client " << poseclntSock->getForeignAddress() << ":" << poseclntSock->getForeignPort()
                 << " connected..." << endl;
        }
    }
    catch (SocketException &e) {
        cerr << e.what() << endl;
        exit(1);
    }
}

void commandHandleClient(TCPSocket *sock) {
    cout << "Command Client " << sock->getForeignAddress() << ":" << sock->getForeignPort() << " connected..." << endl;
    // Send received string and receive again until the end of transmission
    char commandBuffer[4];
    int recvMsgSize;
    try {
        while ((recvMsgSize = sock->recv(commandBuffer, 4)) > 0) { // Zero means end of transmission
            switch (atoi(commandBuffer)) {
                case 9:
                    track_init_state = true;
                    tracking_state = false;
                    target_search_state = false;
                    break;
                default:
                    break;
            }
        }
    }
    catch (SocketException e) {
        cerr << "commandHandleClient:Send Failed...client may be exit." << endl;
        delete sock;
    }
}

void CommandServer() {
    op::log("Command Server Thread Start...Port:9997");
    unsigned short commandServPort = 9997;    // First arg:  local port

    try {
        TCPServerSocket commandServSock(commandServPort);   // Socket descriptor for server
        for (;;) {      // Run forever
            // Create separate memory for client argument

            TCPSocket *clntSock = commandServSock.accept();
            std::thread commandHandleClientThread(commandHandleClient, clntSock);
            commandHandleClientThread.detach();
        }
    }
    catch (SocketException &e) {
        cerr << e.what() << endl;
        exit(1);
    }
}

bool isTargetCoveredwithFace(op::Array<float> &keypoints, std::vector <op::Rectangle<float>> &faceRectangles,
                             op::Rectangle<float> bbox, int target) {
    float ratio_threshold1 = 0.22f;
    //float ratio_threshold2 = 0.32f;
    int numberPeople = faceRectangles.size();
    //printf("numberPeople = %d\n",numberPeople);

    //const auto numberBodyParts = keypoints.getSize(1);
    //auto targetRectangle = op::getKeypointsRectangle(keypoints, target, numberBodyParts, 0.1f);
    //auto targetAverageScore = op::getAverageScore(keypoints, target);
    //printf(" targetAverageScore=%f\n", targetAverageScore);
    for (auto person = 0; person < numberPeople; person++) {
        if (person == target || faceRectangles.at(person).area() < 9000) continue;
        //auto personRectangle = op::getKeypointsRectangle(keypoints, person, numberBodyParts, 0.1f);
        //float body_ratio = tp::bboxOverlapRatio(personRectangle, targetRectangle, tp::RationType::Min);

        float face_ratio = tp::bboxOverlapRatio(faceRectangles.at(person), bbox, tp::RationType::Min);

        //printf(" face_ratio=%f\n", face_ratio);
        //printf(" body_ratio=%f\n", body_ratio);

        if (face_ratio > ratio_threshold1 /*&& targetAverageScore < 0.78*/) {
            return true;
        }

    }
    return false;
}

double
pickMaxFaceSimilarity(cv::Mat &curr_frame, std::vector <op::Rectangle<float>> &faceRectangles, cv::Mat &targetHeadCrop,
                      int &target, RegressorBase *regressor) {
    double similarity_prev = 0.0f;
    //printf("personIndex=%d\n",personIndex);
    int numberPeople = faceRectangles.size();
    cv::Rect rect_crop;
    //printf("numberPeople = %d\n",numberPeople);
    for (auto person = 0; person < numberPeople; person++) {
        if (faceRectangles.at(person).x <= 0 || faceRectangles.at(person).y <= 0 ||
            (faceRectangles.at(person).x + faceRectangles.at(person).width) >= 1280 ||
            (faceRectangles.at(person).y + faceRectangles.at(person).height) >= 720 ||
            faceRectangles.at(person).width < 1 || faceRectangles.at(person).height < 1) {
            continue;
        };
        rect_crop.x = faceRectangles.at(person).x + faceRectangles.at(person).width / 4;
        rect_crop.y = faceRectangles.at(person).y;
        rect_crop.width = faceRectangles.at(person).width / 4 * 2;
        rect_crop.height = faceRectangles.at(person).height / 4 * 3;
        //printf("%d,%d,%d,%d\n", rect_crop.x, rect_crop.y, rect_crop.width, rect_crop.height);
        //printf("%f,%f,%f,%f\n", faceRectangles.at(person).x, faceRectangles.at(person).y, faceRectangles.at(person).width, faceRectangles.at(person).height);
        double similarity = 0.f;
        BoundingBox search_result;
        cv::Mat roi_face(curr_frame, rect_crop);
        cv::resize(roi_face, roi_face, cv::Size(227, 227), (0, 0), (0, 0), cv::INTER_LINEAR);
        regressor->Regress(curr_frame, roi_face, targetHeadCrop, &search_result);
        similarity = search_result.p_;
        if (similarity > similarity_prev) {
            target = person;
            similarity_prev = similarity;
        }
        // printf("find personIndex=%d\n",personIndex);
    }
    return similarity_prev;

}