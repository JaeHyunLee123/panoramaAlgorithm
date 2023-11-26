#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>
#include "blend.h"

using namespace std;
using namespace cv;

Mat stitch_two_image(Mat original_image, Mat object_image);
bool compareX(const Point& p1, const Point& p2);

int main()
{
	vector<Mat> frames;
	VideoCapture video("panorama_video_sampe2.mp4");

	int totalFrames = video.get(CAP_PROP_FRAME_COUNT);
	float skippingFrames = totalFrames / 19.0;

	if (skippingFrames == 0) skippingFrames = 1;

	cout << "Total frames:" << totalFrames << endl;

	for (float i = 0; i < totalFrames; i += skippingFrames) {
		Mat temp;
		video.set(CAP_PROP_POS_FRAMES, static_cast<int>(i));

		video >> temp;
		frames.push_back(temp);

		if (static_cast<int>(i) % 10 == 0) cout << "Loading video :" << static_cast<int>(i) << "/" << totalFrames << endl;
	}


	for (int i = 0; i < frames.size(); i++) {
		resize(frames[i], frames[i], Size(0, 0), 0.5, 0.5, INTER_LINEAR);
	}

	float fixedRow = frames[0].rows;
	int newCol;

	Mat result = stitch_two_image(frames[10], frames[11]);
	newCol = (float)result.cols / (float)result.rows * fixedRow;
	resize(result, result, Size(newCol, fixedRow));

	flip(result, result, 1);
	flip(frames[9], frames[9], 1);
	result = stitch_two_image(result, frames[9]);
	flip(result, result, 1);
	newCol = (float)result.cols / (float)result.rows * fixedRow;
	resize(result, result, Size(newCol, fixedRow));

	/*flip(frames[9], frames[9], 1);
	imshow("1", frames[9]);
	imshow("2", frames[10]);
	imshow("3", frames[11]);*/

	imshow("result", result);
	waitKey(0);

	return 0;
}



Mat stitch_two_image(Mat original_image, Mat object_image) {
	//객체 매칭 시 두 이미지의 크기를 똑같이 해서 잘못된 매칭의 수를 줄이려고 함
	Mat originalCutImage(object_image.size(), CV_8UC3);
	original_image(Rect(original_image.cols - originalCutImage.cols, 0, originalCutImage.cols, originalCutImage.rows)).
		copyTo(originalCutImage(Rect(0, 0, originalCutImage.cols, originalCutImage.rows)));

	//이미지를 내려서 변환할 때 100으로 모자랄 수 있기에 추후 쉽게 수정하기 위해 상수로 만듬
	int SHIFT = original_image.rows / 3;

	// ORB 특징점 검출기 초기화
	int minHessian = 1000;
	cv::Ptr<cv::ORB> orb = cv::ORB::create(minHessian);

	// 특징점 및 기술자 추출
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	orb->detectAndCompute(originalCutImage, noArray(), keypoints1, descriptors1);
	orb->detectAndCompute(object_image, noArray(), keypoints2, descriptors2);


	//-- Step 2: Matching descriptor vectors with a brute force matcher
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

	//좋은 매칭점 선택
	//1.Filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.8f;
	std::vector<DMatch> first_good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i].size() == 2) {
			if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			{
				first_good_matches.push_back(knn_matches[i][0]);
			}
		}
	}


	//2. gradient로 나누기
	// 객체 위치 찾기
	std::vector<cv::Point2f> src_pts, dst_pts;
	int gradientValue[11] = { 0, }; // -0.5~ 0.5로 0.1씩
	std::vector<std::vector<int>> gradientIndex(11); //해당하는 인덱스 저장
	float tmp = 0.0f;
	int tmpIndex = 0;

	for (int i = 0; i < first_good_matches.size(); i++) {
		src_pts.push_back(keypoints1[first_good_matches[i].queryIdx].pt);
		dst_pts.push_back(keypoints2[first_good_matches[i].trainIdx].pt);

		//gradient 계산 및 저장
		tmp = ((float)(dst_pts[i].y - src_pts[i].y)) / ((float)(dst_pts[i].x - src_pts[i].x));
		tmpIndex = (int)(tmp * 10 + 5);
		if (tmpIndex >= 0 && tmpIndex < 11) {
			gradientValue[tmpIndex] = gradientValue[tmpIndex] + 1;
			gradientIndex[tmpIndex].push_back(i);
		}

	}

	//가장 큰 값이 저장된 index만 살리고 나머지는 버린다.
	int maxIndex = 0;
	for (int i = 1; i < 11; i++) {
		if (gradientValue[maxIndex] < gradientValue[i])
			maxIndex = i;
	}

	std::vector<DMatch> second_good_matches;
	for (int i = 0; i < gradientValue[maxIndex]; i++) {
		second_good_matches.push_back(first_good_matches[gradientIndex[maxIndex][i]]);
	}

	//3. 최단 거리를 구해 오차 제거(물리적 거리)
	std::vector<cv::DMatch> third_good_matches;
	double distanceMean = 0;
	double distanceTmp = 0;
	Point2f oriPoint, objPoint;
	for (int i = 0; i < second_good_matches.size(); i++) {
		oriPoint = keypoints1[second_good_matches[i].queryIdx].pt;
		objPoint = keypoints2[second_good_matches[i].trainIdx].pt;

		//거리 공식 사용
		distanceTmp = sqrt(pow(oriPoint.x - objPoint.x - originalCutImage.cols, 2) + pow(oriPoint.y - objPoint.y, 2));
		distanceMean += distanceTmp;
	}

	//평균의 +-20퍼센트 정도만 남기고 삭제
	distanceMean /= second_good_matches.size();

	for (int i = 0; i < second_good_matches.size(); i++) {
		oriPoint = keypoints1[second_good_matches[i].queryIdx].pt;
		objPoint = keypoints2[second_good_matches[i].trainIdx].pt;

		//거리 공식 사용
		distanceTmp = sqrt(pow(oriPoint.x - objPoint.x - originalCutImage.cols, 2) + pow(oriPoint.y - objPoint.y, 2));
		if (distanceTmp < distanceMean * 1.1 && distanceTmp > distanceMean * 0.9)
			third_good_matches.push_back(second_good_matches[i]);
	}

	// 좋은 매칭으로 객체 위치 찾기
	//매칭 결과가 너무 작으면 호모그래피를 찾을 수 없기에 이를 막기 위해 코드 추가
	std::vector<cv::Point2f> newSrc_pts, newDst_pts;
	if (third_good_matches.size() > 4) {
		for (int i = 0; i < third_good_matches.size(); i++) {
			newSrc_pts.push_back(keypoints1[third_good_matches[i].queryIdx].pt);
			newDst_pts.push_back(keypoints2[third_good_matches[i].trainIdx].pt);
		}
	}
	else {
		for (int i = 0; i < second_good_matches.size(); i++) {
			newSrc_pts.push_back(keypoints1[second_good_matches[i].queryIdx].pt);
			newDst_pts.push_back(keypoints2[second_good_matches[i].trainIdx].pt);
		}
	}



	// 변환 행렬 계산 -> CV_64F
	Mat H = findHomography(newDst_pts, newSrc_pts, cv::RANSAC);

	// 변환 행렬을 수정해서 0,100의 translate를 적용한다.
	// 100이라는 수치는 추후 조정할 수 있다. 100을 해도 이미지가 잘리는 경우가 존재할 수 있기 때문
	double set[9] = { 1,0, 0, 0, 1, SHIFT, 0, 0, 1 };
	Mat translateShift = Mat(H.size(), CV_64F, set);
	Mat translateH = translateShift * H;


	// 변환행렬을 적용해 object_on_original에 저장 
	Mat object_on_original;
	cv::warpPerspective(object_image, object_on_original, translateH, Size(object_image.cols * 2, object_image.rows * 2), INTER_CUBIC);
	//imshow("object_on_original", object_on_original);

	// 블랜딩을 위해 object_on_original의 경계 부분의 좌표를 구할 필요가 있다.
	// 외곽선을 구해주는 함수를 이용해서 이를 계산하겠다.
	Mat contoursMask;
	cvtColor(object_on_original, contoursMask, COLOR_BGR2GRAY); // RGB -> GRAY (채널 문제 때문);


	//이미지의 경계 찾기
	//contours에 외곽선의 좌표 저장
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(contoursMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


	//contours에서 각 외곽선에 대해 꼭짓점을 찾습니다.
	//꼭짓점은 시작점을 기준으로 시계방향으로 저장된다.
	vector<vector<Point>> approxContours(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		double epsilon = 0.02 * arcLength(contours[i], true);
		approxPolyDP(contours[i], approxContours[i], epsilon, true);
	}


	// 좌표를 x값이 큰 순서대로 정렬 후 저장한다.
	// 오른쪽에서 col이 큰 포인트 순서대로 저장
	//이때 contours의 인덱스가 1이 아닌 경우도 있을 수 있으므로 크기가 가장 큰 index를 기준으로 해당 과정을 수행한다.
	Point rightMaxPoint, rightMinPoint, leftMaxPoint, leftMinPoint;
	int contoursIndex = 0;
	for (contoursIndex = 0; contoursIndex < approxContours.size(); contoursIndex++) {
		if (approxContours[contoursIndex].size() == 4) {
			sort(approxContours[contoursIndex].begin(), approxContours[contoursIndex].end(), compareX);
			leftMinPoint = approxContours[contoursIndex][0];
			leftMaxPoint = approxContours[contoursIndex][1];
			rightMinPoint = approxContours[contoursIndex][2];
			rightMaxPoint = approxContours[contoursIndex][3];
		}
	}
	contoursIndex--;

	//양쪽 좌표 중 y값의 크기를 비교하고 결과를 저장한다.
	bool isLeftYMax, isRightYMax;

	if (leftMinPoint.y < leftMaxPoint.y)
		isLeftYMax = true;
	else
		isLeftYMax = false;

	if (rightMinPoint.y < rightMaxPoint.y)
		isRightYMax = true;
	else
		isRightYMax = false;


	//이미지를 붙이고 필터를 붙이는 방식으로 구현하자
	Mat preCombineImg = Mat::zeros(object_on_original.size(), CV_8UC3);

	//먼저 originalCutImage를 붙인다.
	originalCutImage(Rect(0, 0, originalCutImage.cols, originalCutImage.rows)).
		copyTo(preCombineImg(Rect(0, SHIFT, originalCutImage.cols, originalCutImage.rows)));

	//object_and_origianl을 붙인다.
	for (int i = 0; i < object_on_original.rows; i++) {
		for (int j = 0; j < object_on_original.cols; j++) {
			//해당 점이 경계선 안에 있는 경우 값 대입
			if (pointPolygonTest(contours[contoursIndex], Point(j, i), false) > 0)
				preCombineImg.at<cv::Vec3b>(i, j) = object_on_original.at<cv::Vec3b>(i, j);
		}
	}
	//imshow("preCombineImg", preCombineImg);

	//이미지를 자르기
	//작은 범위로 자르기
	int top, right;
	if (isRightYMax) {
		top = max(SHIFT, rightMinPoint.y);
	}
	else {
		top = max(SHIFT, rightMaxPoint.y);
	}
	right = min(rightMaxPoint.x, rightMinPoint.x);

	//blending을 위한 경계선 벡터 생성
	vector<int> center;
	for (int i = top; i < top + original_image.rows; i++) {
		for (int j = 0; j < object_on_original.cols; j++) {
			//해당 점이 경계선 위에 있는 경우 값 넣기
			if (pointPolygonTest(contours[contoursIndex], Point(j, i), false) == 0)
			{
				//결과 영상에 대해 수행할 예정이기에 original_image를 붙인만큼의 길이를 계산해줘야 한다.
				center.push_back(j + original_image.cols - originalCutImage.cols);
				break;
			}
		}
	}


	// 결과를 저장할 mat 생성 후 데이터 옮기기
	//잘린 original과 object_on_original를 합쳐야 한다.
	//분기문을 넣은 이유는 original_image.cols - originalCutImage.cols가 0이면 오류가 발생하기 때문이다.
	Mat result = Mat::zeros(original_image.rows, original_image.cols - originalCutImage.cols + right, CV_8UC3);

	if (original_image.cols - originalCutImage.cols == 0) {
		preCombineImg(Rect(0, top, right, result.rows)).
			copyTo(result(Rect(0, 0, right, result.rows)));
	}
	else {
		original_image(Rect(0, 0, original_image.cols - originalCutImage.cols, result.rows)).
			copyTo(result(Rect(0, 0, original_image.cols - originalCutImage.cols, result.rows)));
		preCombineImg(Rect(0, top, right, result.rows)).
			copyTo(result(Rect(original_image.cols - originalCutImage.cols, 0, right, result.rows)));
	}

	//blending 수행
	result = ljh::blendImage(result, center, 20, 2);
	return result;
}

bool compareX(const Point& p1, const Point& p2) {
	return p1.x < p2.x;
}
