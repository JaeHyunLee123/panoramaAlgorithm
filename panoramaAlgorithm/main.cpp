#include <opencv2/opencv.hpp>
#include <iostream>
#include <array>
#include <cmath>
#include <vector>
#include <iomanip>
#include "blend.h"

using namespace std;
using namespace cv;

//akaze 탐지를 위한 상수
const float inlier_threshold = 2.5f; // Distance threshold to identify inliers with homography check


const double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
const int bb_min_inliers = 100; // Minimal number of inliers to draw bounding box
const int stats_update_period = 10; // On-screen statistics are updated every 10 frames

void akazeBF();

int main()
{
    VideoCapture video("panorama_video_sampe2.mp4");

    Mat frame1, frame2;

    int totalFrames = video.get(CAP_PROP_FRAME_COUNT);
    float skippingFrames = totalFrames / 20.0;

    


    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;
    Ptr<AKAZE> akaze = AKAZE::create();
    Ptr<ORB> orb = ORB::create(minHessian);
    Ptr<SIFT> sift = SIFT::create(minHessian);
    std::vector<KeyPoint> keypoints1;
    std::vector<KeyPoint> keypoints2;
    Mat descriptors1, descriptors2;

    for (int i = 0; i < totalFrames; i += skippingFrames) {
        cout << "Loading..." << endl;
        video.set(CAP_PROP_POS_FRAMES, i);
        video >> frame1;

        video.set(CAP_PROP_POS_FRAMES,  i);
        video >> frame2;

        orb->detectAndCompute(frame1, noArray(), keypoints1, descriptors1);
        orb->detectAndCompute(frame2, noArray(), keypoints2, descriptors2);

        //-- Step 2: Matching descriptor vectors with a brute force matcher
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.75f;
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        //-- Draw keypoints
        Mat img_keypoints1;
        drawKeypoints(frame1, keypoints1, img_keypoints1);
        Mat img_keypoints2;
        drawKeypoints(frame2, keypoints2, img_keypoints2);
        Mat img_matches;
        drawMatches(frame1, keypoints1, frame2, keypoints2, good_matches, img_matches);


        cout << "Frame: " << i << "/" << totalFrames << endl;
    
        //-- Show detected matches
        imshow("Matches", img_matches);
        cv::waitKey(0);
    
    }


    return 0;
}

//SIFT + BF (기존 방식)

//AKAZE + BF
void akazeBF() {
    VideoCapture video("panorama_video_sampe2.mp4");

    Mat frame1, frame2;

    int totalFrames = video.get(CAP_PROP_FRAME_COUNT);
    float skippingFrames = totalFrames / 20.0;

    video.set(CAP_PROP_POS_FRAMES, skippingFrames * 10);
    video >> frame1;

    video.set(CAP_PROP_POS_FRAMES, skippingFrames * 11);
    video >> frame2;

    Mat homography;
    FileStorage fs(samples::findFile("H1to3p.xml"), FileStorage::READ);
    fs.getFirstTopLevelNode() >> homography;
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(frame1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(frame2, noArray(), kpts2, desc2);
    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);
    vector<KeyPoint> matched1, matched2;
    for (size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        if (dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }
    vector<DMatch> good_matches;
    vector<KeyPoint> inliers1, inliers2;
    for (size_t i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;
        col = homography * col;
        col /= col.at<double>(2);
        double dist = sqrt(pow(col.at<double>(0) - matched2[i].pt.x, 2) +
            pow(col.at<double>(1) - matched2[i].pt.y, 2));
        if (dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    Mat res;
    drawMatches(frame1, inliers1, frame2, inliers2, good_matches, res);

    imshow("result", res);

    waitKey(0);
}

//AKAZE + FLANN
