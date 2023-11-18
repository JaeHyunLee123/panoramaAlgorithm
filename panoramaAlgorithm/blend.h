#pragma once
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

namespace ljh {
	struct RGB {
		int red;
		int green;
		int blue;
	};

	struct HSI
	{
		int hue; // 0 ~ 180/PI 사이 값
		int saturation; //0 ~ 100
		int intensity; // 0 ~ 255
	};

	//mergedImage: blending 대상 이미지
	//center: center 점 모음
	//blendingArea: blending 하는 영역
	//errorRange: 오차범위
	Mat blendImage(Mat image, vector<int> center, int blendingArea, int errorRange);

	vector<RGB> blendRow(vector<RGB> input, int center, int blendingArea, int errorRange);

	RGB getRGB(Mat image, int col, int row);
	void putRGB(Mat& dest, RGB rgb, int col, int row);

	RGB hsi2rgb(HSI input);
	HSI rgb2hsi(RGB input);
}