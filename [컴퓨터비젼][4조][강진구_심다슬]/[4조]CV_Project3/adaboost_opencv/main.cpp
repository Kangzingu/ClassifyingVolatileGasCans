#include "AdaBooster.h"
#include "CrossValidator.h"
#include "FeatureParser.h"
#include "FeatureVector.h"
#include "sorting.h"
#include "StrongClassifier.h"
#include "TrainingData.h"
#include "WeakClassifier.h"
#include <fstream>
#include<stdio.h>
#include<iostream>

using namespace std;
#include <string>
#include "opencv2/opencv.hpp"
using namespace cv;

//이미지 관련
#define   WIDTH   128
#define HEIGHT   128
#define   ORIGINAL_WINDOW_SIZE   24

#define CLASSIFIERNUM 100
#define TR_IMG_PER_CLASS   60

#define	TR_POS_DIR "E:/positive"
#define	TR_NEG_DIR "E:/negative"

TrainingData td;
FeatureParser fp;
AdaBooster adaboost;

vector<float> extractFeature(Mat integralImg);
Mat calculateIntegralFrom(Mat input);
float calMaskfromIntegral(Mat integral, int x, int y, int w, int h);

typedef struct {
	int x; //시작 x좌표
	int y; //시작 y좌표
	int w; //너비
	int h; //높이
	int cx;
	int cy;
	int rank;
	float conf; //confidence
	float error;
	int id;
	//   int sx; //step-size x
	//   int sy; //step-size y
}Block;

typedef struct {
	Block* comb;
	float* weights;
	int numCom;
}MSHF;

//////////////////////


int main() {
	//cout << "코드 수정완료" << endl;

	/*
	1. feature 입력

	*/

	////positive 영상 입력 => 특징 추출
	char filepath[300];

	vector<pair<vector<float>, int>> trData;
	for (int i = 1; i < TR_IMG_PER_CLASS + 1; i++)
		//int i = 1;
	{
		sprintf_s(filepath, "%s/%d.bmp", TR_POS_DIR, i);
		cout << filepath << endl;

		Mat img = imread(filepath, CV_LOAD_IMAGE_GRAYSCALE);
		Mat mat = calculateIntegralFrom(imread(filepath, CV_LOAD_IMAGE_GRAYSCALE));
		vector<float> feature = extractFeature(mat);
		cout << "feature size" << feature.size() << endl;
		int label = 1;
		trData.push_back(make_pair(feature, label));
	}

	cout << endl << "positive 완료" << endl;

	////negative 영상 입력 => 특징 추출
	//for (int i = 1; i < TR_IMG_PER_CLASS + 1; i++)
	//{
	//	sprintf_s(filepath, "%s/%d.bmp", TR_POS_DIR, i);
	//	cout << filepath << endl;
	//	Mat mat = calculateIntegralFrom(imread(filepath, CV_LOAD_IMAGE_GRAYSCALE));
	//	vector<float> feature = extractFeature(mat);
	//	
	//	int label = -1;
	//	trData.push_back(make_pair(feature, label));
	//}
	//cout << endl << "negative 완료" << endl;

	int featDim = trData[0].first.size();
	vector<float> ftrs;

	for (int i = 0; i < trData.size(); i++) {
		FeatureVector fv(trData[i].first, trData[i].second);	
		if (!td.addFeature(fv))
			printf("ERROR: feature vector incorrect size!\n");
	}


	ofstream outFile("result.csv", ios::app);
	outFile.write();//결과 넣어줌
	outFile.close();//파일닫기

	
	//최종 main 함수
	//AdaBooster adaboost;
	//adaboost.getStrongClassifier(td, CLASSIFIERNUM);

	int x;
	cin >> x;
	return 0;
}


Mat calculateIntegralFrom(Mat input)
{
	Mat integralimg(Size(input.cols + 1, input.rows + 1), CV_32F); //integral image는 연산 특성상 사이즈가 1픽셀씩 증가한다.

	for (int i = 0; i < integralimg.rows; i++)
		for (int j = 0; j < integralimg.cols; j++)
		{
			if (i == 0 || j == 0)
				integralimg.at<float>(i, j) = 0;
			else
				integralimg.at<float>(i, j) = input.at<unsigned char>(i - 1, j - 1);
		}

	for (int i = 0; i < input.rows + 1; i++)
		for (int j = 0; j < input.cols + 1; j++)
		{
			if (i == 0 && j == 0)
			{
				continue;
			}
			else if (i == 0)
			{
				integralimg.at<float>(i, j) += integralimg.at<float>(i, j - 1);
			}
			else if (j == 0)
			{
				integralimg.at<float>(i, j) += integralimg.at<float>(i - 1, j);
			}
			else
			{
				integralimg.at<float>(i, j) += (integralimg.at<float>(i - 1, j) +
					integralimg.at<float>(i, j - 1) - integralimg.at<float>(i - 1, j - 1));
			}
		}
	return integralimg;
}

void drawCombineMSHF(MSHF m, char* filename)
{
	IplImage* confMapImg = cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 3);
	for (int i = 0; i < confMapImg->height; i++)
		for (int j = 0; j < confMapImg->width; j++)
		{
			confMapImg->imageData[(i * confMapImg->width + j) * 3 + 2] = (unsigned char)0;
			confMapImg->imageData[(i * confMapImg->width + j) * 3 + 1] = (unsigned char)0;
			confMapImg->imageData[(i * confMapImg->width + j) * 3] = (unsigned char)0;
		}

	for (int i = 0; i < m.numCom; i++)
	{
		if (m.weights[i] > 0)
		{
			cvRectangle(confMapImg, CvPoint(m.comb[i].x + 1, m.comb[i].y + 1),
				CvPoint(m.comb[i].x + m.comb[i].w - 1, m.comb[i].y + m.comb[i].h - 1),
				Scalar(0, 0, 255), 3, 8, 0);
		}
		else
		{
			cvRectangle(confMapImg, CvPoint(m.comb[i].x + 1, m.comb[i].y + 1),
				CvPoint(m.comb[i].x + m.comb[i].w - 1, m.comb[i].y + m.comb[i].h - 1),
				Scalar(255, 0, 0), 3, 8, 0);
		}
	}

	cvSaveImage(filename, confMapImg);
	cvReleaseImage(&confMapImg);
}

vector<float> extractFeature(Mat integralImg)
{
	vector<float> vec;
	int featType[5][2] = { { 1, 2 }, { 1, 3 }, { 2, 1 }, { 3, 1 }, { 2, 2 } };
	int windowScale = integralImg.rows / ORIGINAL_WINDOW_SIZE;
	int imageH = integralImg.rows;
	int imageW = integralImg.cols;
	//imageH 25 imageW25  windowScale 1
	//cout <<"Log: "<< imageH << " " << imageW << " " << windowScale << endl;  
	int maxH = imageH / windowScale;
	int maxW = imageW / windowScale;
	int count = 0;
	int index = 0;

	for (int type = 0; type < 5; type++) {
		int windowCountH = featType[type][0];
		int windowCountW = featType[type][1];
		for (int featH = 1; featH <= (maxH / windowCountH); featH++) {
			for (int featW = 1; featW <= (maxW / windowCountW); featW++) {
				for (int y = 0; y < imageH - (featH * windowScale * windowCountH - 1); y += windowScale) {
					for (int x = 0; x < imageW - (featW * windowScale * windowCountW - 1); x += windowScale) {
						//cout << index << endl;
						int width = featW * windowScale;
						int height = featH * windowScale;
						float result = 0;
						// type 1
						if (windowCountH == 1 && windowCountW == 2) {
							int sum1 = calMaskfromIntegral(integralImg, x, y, width, height);
							int sum2 = calMaskfromIntegral(integralImg, x + width, y, width, height);
							result = sum1 - sum2;
						}
						// type 2
						else if (windowCountH == 1 && windowCountW == 3) {
							int sum1 = calMaskfromIntegral(integralImg, x, y, width, height);
							int sum2 = calMaskfromIntegral(integralImg, x + width, y, width, height);
							int sum3 = calMaskfromIntegral(integralImg, x + width * 2, y, width, height);
							result = sum1 - sum2 + sum3;
						}
						// type 3
						else if (windowCountH == 2 && windowCountW == 1) {
							int sum1 = calMaskfromIntegral(integralImg, x, y, width, height);
							int sum2 = calMaskfromIntegral(integralImg, x, y + height, width, height);
							result = sum1 - sum2;
						}
						// type 4
						else if (windowCountH == 3 && windowCountW == 1) {
							int sum1 = calMaskfromIntegral(integralImg, x, y, width, height);
							int sum2 = calMaskfromIntegral(integralImg, x, y + height, width, height);
							int sum3 = calMaskfromIntegral(integralImg, x, y + height * 2, width, height);
							result = sum1 - sum2 + sum3;
						}
						// type 5
						else {
							int sum1 = calMaskfromIntegral(integralImg, x, y, width, height);
							int sum2 = calMaskfromIntegral(integralImg, x, y + height, width, height);
							int sum3 = calMaskfromIntegral(integralImg, x + width, y, width, height);
							int sum4 = calMaskfromIntegral(integralImg, x + width, y + height, width, height);
							result = sum1 - sum2 - sum3 + sum4;
						}
						//count ++ ;
					 //   cout << count << endl;
					 //   cout << endl << "result: " << result << endl;

						vec.push_back(result);

					}
				}
			}
		}
	}
	cout << "size " << vec.size() << endl;
	return vec;
}

float calMaskfromIntegral(Mat integral, int x, int y, int w, int h)
{
	//cout << "integral" << endl;
	int endX = x + w > ORIGINAL_WINDOW_SIZE ? 23 : x + w;
	int endY = y + h > ORIGINAL_WINDOW_SIZE ? 23 : y + h;
	int sum = integral.at<float>(endY, endX);
	sum -= x > 0 ? integral.at<float>(endY, x - 1) : 0;
	sum -= y > 0 ? integral.at<float>(y - 1, endX) : 0;
	sum += x > 0 && y > 0 ? integral.at<float>(y - 1, x - 1) : 0;
	//cout << "sum " <<sum << endl;
	return sum;
}