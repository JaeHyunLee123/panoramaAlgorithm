#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include "blend.h"

using namespace std;
using namespace cv;

namespace ljh {
	vector<RGB> blendRow(vector<RGB> input, int center, int blendingArea, int errorRange) {
		vector<RGB> result(input);

		//오차 범위 내 처리
		for (int i = 0; i < blendingArea / 2 - errorRange / 2; i++) {
			//center를 기준으로 대칭되는 픽셀의 값을 가져온다
			int leftIndex = center - blendingArea / 2 + i;
			int rightIndex = center + blendingArea / 2 - i;

			RGB leftRGB = input[leftIndex];
			RGB rightRGB = input[rightIndex];

			//가져온 픽셀 값을 hsi 값으로 바꾼다
			HSI leftHSI = rgb2hsi(leftRGB);
			HSI rightHSI = rgb2hsi(rightRGB);

			//intensity 값으로 알파 블렌딩을 진행한다.
			double alpha = (double)i / blendingArea;

			int tempIntensity = leftHSI.intensity;
			leftHSI.intensity = (1 - alpha) * leftHSI.intensity + alpha * rightHSI.intensity;
			rightHSI.intensity = alpha * tempIntensity + (1 - alpha) * rightHSI.intensity;


			//다시 rgb 값으로 바꾼 후 저장
			leftRGB = hsi2rgb(leftHSI);
			rightRGB = hsi2rgb(rightHSI);

			result[leftIndex] = leftRGB;
			result[rightIndex] = rightRGB;
		}

		//오차범위 바로 바깥의 intensity 값 평균으로 오차범위 내 intensity 조정
		RGB leftStandardRGB = result[center - errorRange / 2];
		RGB rightStandardRGB = result[center + errorRange / 2];

		HSI leftStandardHSI = rgb2hsi(leftStandardRGB);
		HSI rightStandardHSI = rgb2hsi(rightStandardRGB);
		int average = (leftStandardHSI.intensity + rightStandardHSI.intensity) / 2;

		for (int i = blendingArea / 2 - errorRange / 2; i <= blendingArea / 2; i++) {
			//center를 기준으로 대칭되는 픽셀의 값을 가져온다
			int leftIndex = center - blendingArea / 2 + i;
			int rightIndex = center + blendingArea / 2 - i;

			RGB leftRGB = result[leftIndex];
			RGB rightRGB = result[rightIndex];

			//가져온 픽셀 값을 hsi 값으로 바꾼다
			HSI leftHSI = rgb2hsi(leftRGB);
			HSI rightHSI = rgb2hsi(rightRGB);

			//오차범위 바로 바깥의 intensity의 평균값 사용
			leftHSI.intensity = average;
			rightHSI.intensity = average;

			//다시 rgb 값으로 바꾼 후 저장
			leftRGB = hsi2rgb(leftHSI);
			rightRGB = hsi2rgb(rightHSI);

			result[leftIndex] = leftRGB;
			result[rightIndex] = rightRGB;
		}
		return result;
	}

	Mat blendImage(Mat image, vector<int> center, int blendingArea, int errorRange) {

		assert(image.rows == center.size());

		Mat result = image.clone();

		for (int row = 0; row < image.rows; row++) {
			//열에 있는 데이터를 vector에 저장한다
			vector<RGB> inputValues = {};
			for (int col = 0; col < image.cols; col++) {
				RGB temp = getRGB(image, col, row);
				inputValues.push_back(temp);
			}
			//저장한 vector로 블렌딩을 진행한다
			vector<RGB> blendedValue = blendRow(inputValues, center[row], blendingArea, errorRange);

			//블렌딩한 값을 result에 넣는다
			for (int col = 0; col < image.cols; col++) {
				putRGB(result, blendedValue[col], col, row);
			}
		}

		return result;
	}

	RGB getRGB(Mat image, int col, int row) {
		RGB result = { 0,0,0 };

		result.red = image.data[(col + row * image.cols) * 3 + 2];
		result.green = image.data[(col + row * image.cols) * 3 + 1];
		result.blue = image.data[(col + row * image.cols) * 3];

		return result;
	}

	void putRGB(Mat& dest, RGB rgb, int col, int row) {
		dest.data[(col + row * dest.cols) * 3 + 2] = rgb.red;
		dest.data[(col + row * dest.cols) * 3 + 1] = rgb.green;
		dest.data[(col + row * dest.cols) * 3] = rgb.blue;
	}

	RGB hsi2rgb(HSI input) {
		RGB result = { 0,0,0 };

		const double PI = 3.141592;

		//계산을 위한 변수 선언. 이 값들은 0~1 사이 값을 가짐. 나중에 255 곱해서 리턴
		double red = 0;
		double green = 0;
		double blue = 0;

		//계산을 위해 normalization
		double hue = input.hue * PI / 180.0;
		double saturation = input.saturation / 100.0;
		double intensity = input.intensity / 255.0;

		//공식 적용
		if (hue < 2 * PI / 3) {
			blue = intensity * (1 - saturation);
			red = intensity * (1 + saturation * cos(hue) / cos(PI / 3 - hue));
			green = 3 * intensity - (red + blue);
		}
		else if (2 * PI / 3 <= hue && hue < 4 * PI / 3) {
			hue -= 2 * PI / 3;

			red = intensity * (1 - saturation);
			green = intensity * (1 + saturation * cos(hue) / cos(PI / 3 - hue));
			blue = 3 * intensity - (red + green);
		}
		else {
			hue -= 4 * PI / 3;

			green = intensity * (1 - saturation);
			blue = intensity * (1 + saturation * cos(hue) / cos(PI / 3 - hue));
			red = 3 * intensity - (blue + green);
		}

		result.red = red * 255;
		result.green = green * 255;
		result.blue = blue * 255;

		return result;
	}

	HSI rgb2hsi(RGB input) {
		HSI result = { 0,0,0 };
		const double PI = 3.141592;

		//계산을 위해 normalization
		double sum = input.red + input.green + input.blue;
		double red = input.red / sum;
		double green = input.green / sum;
		double blue = input.blue / sum;

		//계산을 위한 변수 선언. 이 값들은 0~1 사이 값을 가짐. 나중에 각각 180/PI와 100을 곱해서 리턴
		double hue = 0;
		double saturation = 0;

		//공식 적용
		double theta = acos(0.5 * ((red - green) + (red - blue)) / pow((pow(red - green, 2) + (red - blue) * (green - blue)), 0.5));
		if (blue > green) {
			hue = 2 * PI - theta;
		}
		else {
			hue = theta;
		}

		saturation = 1 - 3 * min(red, min(green, blue));

		result.hue = hue * 180 / PI;
		result.saturation = saturation * 100;
		result.intensity = (input.red + input.green + input.blue) / 3;

		return result;
	}
}