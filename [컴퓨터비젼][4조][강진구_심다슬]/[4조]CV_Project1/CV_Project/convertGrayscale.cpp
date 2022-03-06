
#pragma once
// etc header
#include <iostream>
#include <fstream>
#include<stdio.h>
#include <string>

//정의한 header
#include "AdaBooster.h"
#include "FeatureVector.h"
#include "sorting.h"
#include "StrongClassifier.h"
#include "TrainingData.h"
#include "WeakClassifier.h"
#include "Element.h"

//opencv header
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

//CLASSIFIER 수
#define CLASSIFIERNUM 20

//이미지 관련
#define WIDTH   128
#define HEIGHT  128
#define   ORIGINAL_WINDOW_SIZE   128//width, height 동일

//학습 데이터 수
#define   TR_IMG_PER_CLASS_POSITIVE 60
#define   TR_IMG_PER_CLASS_NEGATIVE 60

//학습 데이터 경로
#define TR_POS_DIR "K:/positive"
#define TR_NEG_DIR "K:/negative"

// 결과 이미지 저장 경로
#define RESULT_IMAGE "K:/resultImage"


//처리 결과 관련(다슬 로그)
#define	DONE 1
#define	FAIL 0

vector<Element> extractFeature(Mat integralImg);
Mat calculateIntegralFrom(Mat input);
float calMaskfromIntegral(Mat integral, int x, int y, int w, int h);
Mat LBP(Mat src_image);

//CART TREE 속성 생성 관련 전역변수
StrongClassifier strongClassifier;
TrainingData td;
AdaBooster adaboost;

vector <int> valList;

int pointCount = 0;

int main() {
	
	Mat mainImage;

	char filepath[300];
	sprintf_s(filepath, "full2.jpg");
	mainImage = imread(filepath);

	cvtColor(mainImage, mainImage, CV_RGB2GRAY);
	//imwrite("K:/positive_grayscale/1.bmp", mainImage);
	imwrite("full2_gray.jpg", mainImage);

	return 0;
}