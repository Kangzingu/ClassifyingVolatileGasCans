
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
#define CLASSIFIERNUM 50

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


//CART TREE 속성 생성 관련 전역변수
StrongClassifier strongClassifier;
TrainingData td;
AdaBooster adaboost;

int main() {

	char filepath[300];
	Mat img;
	vector<pair<vector<Element>, int>> trData;

	//positive 영상 입력 => 특징 추출
	for (int i = 1; i < TR_IMG_PER_CLASS_POSITIVE + 1; i++)

	{
		sprintf_s(filepath, "%s/%d.bmp", TR_POS_DIR, i);
		cout << filepath << endl;

		////(진구) : LBP feature 생성으로 변경 필요
		Mat mat = calculateIntegralFrom(imread(filepath));
		vector<Element> feature = extractFeature(mat);
		////

		//posive label==1
		int label = 1;

		trData.push_back(make_pair(feature, label));



	}

	cout << endl << "positive 완료" << endl;



	//negative 영상 입력 => 특징 추출
	for (int i = 1; i < TR_IMG_PER_CLASS_NEGATIVE + 1; i++)
	{

		sprintf_s(filepath, "%s/%d.bmp", TR_NEG_DIR, i);
		cout << filepath << endl;

		////(진구) : LBP feature 생성으로 변경 필요
		Mat mat = calculateIntegralFrom(imread(filepath));
		vector<Element> feature = extractFeature(mat);
		////

		//negative label==-1
		int label = -1;
		trData.push_back(make_pair(feature, label));

	}

	cout << endl << "negative 완료" << endl;


	///////////////////////////////////////////////
	/*for (int i = 0; i < trData.size(); i++) {
	int featDim = trData[i].first.size();
	int label = trData[i].second;
	printf("featDim: %d, label: %d\n", featDim, label);
	}*/



	vector<float> ftrs;


	for (int i = 0; i < trData.size(); i++) {

		FeatureVector fv(trData[i].first, trData[i].second);
		if (!td.addFeature(fv))
			printf("ERROR: feature vector incorrect size!\n");
	}

	/*
	feature 생성 확인

	: 임의의 이미지에 박스를 표시하여, feautre가 convolution 형태로 제대로 뽑히는 지 확인
	*/


	//vector<Element> tmpfv = trData[0].first;
	//for (int i = 0; i <trData[0].first.size(); i++) {
	//	char ofilename[200];
	//	Mat draw_img=img.clone();
	//	
	//	rectangle(draw_img, Point(tmpfv.at(i).start_x, tmpfv.at(i).start_y),
	//		Point(tmpfv.at(i).end_x, tmpfv.at(i).end_y), Scalar(0, 0, 255));
	//	
	//	/*rectangle(draw_img, Point(10,10),
	//		Point(20,20), Scalar(0, 0, 255));*/
	//	cout <<"i:" <<i<<" sx: " << tmpfv.at(i).start_x << " sy: " << tmpfv.at(i).start_y << " ex: " << tmpfv.at(i).end_x << " ey: " << tmpfv.at(i).end_y << endl;
	//	sprintf_s(ofilename, "%s\\%d.bmp", SAVE_IMAGE, i);
	////	imshow("combine_img", draw_img);
	//	imwrite(ofilename, draw_img);
	//}

	//
	//waitKey();


	/*
	AdaBoost 학습
	*/
	AdaBooster adaboost;
	strongClassifier = adaboost.getStrongClassifier(td, CLASSIFIERNUM);
	////학습 종료


	vector<WeakClassifier>weakList = strongClassifier.weakClassifiers();
	//i번째 weakset의 start_x, start_y, end_x, end_y, acc 출력

	cout << endl;
	for (int i = 0; i < weakList.size(); i++) {

		weakList.at(i).printClassifier();
		weakList.at(i).writeClassifier("weakClassifier_info.txt");
	}
	cout << endl;

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

vector<Element> extractFeature(Mat integralImg)
{

	vector<Element> vec;
	int featType[5][2] = { { 1, 2 },{ 1, 3 },{ 2, 1 },{ 3, 1 },{ 2, 2 } };
	//int windowScale_X = integralImg.cols / ORIGINAL_WINDOW_SIZE_X;
	//int windowScale_Y = integralImg.rows / ORIGINAL_WINDOW_SIZE_Y;
	int windowScale_X;
	int windowScale_Y;

	//int windoScale = integralImg.cols  ;
	//int windowScale = 1;
	int imageH = integralImg.rows;
	int imageW = integralImg.cols;
	//imageH 25 imageW25  windowScale 1
	//	cout <<"Log: "<< imageH << " " << imageW << " " << windowScale << endl;  
	int maxH;
	int maxW;
	//int maxH = imageH;
	//int maxW = imageW;
	int count = 0;
	int index = 0;
	int n = 0;

	//text파일 읽는 코드

	//1. 파일 읽기
	ifstream fin("variable_block.txt");

	if (!fin)
	{
		cerr << "[variable_block txt file read]fail" << endl;
		//return -1;
	}

	int a, b, c, d;

	while (!fin.eof()) {
		fin >> a >> b >> c >> d;
		/*cout << endl<<a <<"\t"<< b << "\t"
		<< c << "\t" << d <<endl;
		*/
		windowScale_X = c;
		windowScale_Y = d;
		maxH = imageH / windowScale_Y;
		maxW = imageW / windowScale_X;
		//a,b,c,d
		//wsx=c; wsu=d;  width=a; height =b;

		int type;
		int width = a;
		int height = b;

		/*for (int type = 0; type < 5; type++) {
		int windowCountH = featType[type][0];
		int windowCountW = featType[type][1];*/
		/*for (int featH = 1; featH <= (maxH / 1); featH++) {
		for (int featW = 1; featW <= (maxW / 1); featW++) {*/
		for (int y = 0; y < imageH - (height); y += windowScale_Y) {
			for (int x = 0; x < imageW - (width); x += windowScale_X) {
				//cout << index << endl;
				//cout << "windowScale_X" << windowScale_X << "windowScale_Y: " << windowScale_Y << endl;
				//int width = featW * windowScale_X;
				//int height = featH * windowScale_Y;



				float result = 0;
				//type 1,3은 사용하지 않음

				// type 1  =>0
				if (width>height) {
					int sum1 = calMaskfromIntegral(integralImg, x, y, width / 2, height);
					int sum2 = calMaskfromIntegral(integralImg, x + width / 2, y, width / 2, height);
					result = sum1 - sum2;
					type = 0;
					//	cout << " x " << x << " y: " << y << " width: " << width << " height: " << height <<endl;
					count++;
				}
				// type 2 =>1 =>사용안함
				/*else if () {
				int sum1 = calMaskfromIntegral(integralImg, x, y, width, height);
				int sum2 = calMaskfromIntegral(integralImg, x + width, y, width, height);
				int sum3 = calMaskfromIntegral(integralImg, x + width * 2, y, width, height);
				result = sum1 - sum2 + sum3;
				}
				*/// type 3 =>2 
				else if (width<height) {
					int sum1 = calMaskfromIntegral(integralImg, x, y, width, height / 2);
					int sum2 = calMaskfromIntegral(integralImg, x, y + height / 2, width, height / 2);
					result = sum1 - sum2;
					type = 2;
					//	cout << " x " << x << " y: " << y << " width: " << width << " height: " << height << endl;
					count++;
				}
				// type 4=>3 =>사용안함
				/*else if (windowCountH == 3 && windowCountW == 1) {
				int sum1 = calMaskfromIntegral(integralImg, x, y, width, height);
				int sum2 = calMaskfromIntegral(integralImg, x, y + height, width, height);
				int sum3 = calMaskfromIntegral(integralImg, x, y + height * 2, width, height);
				result = sum1 - sum2 + sum3;
				}*/
				// type 5 =>4
				else if (width = height) {
					int sum1 = calMaskfromIntegral(integralImg, x, y, width / 2, height / 2);
					int sum2 = calMaskfromIntegral(integralImg, x + width / 2, y, width / 2, height / 2);
					int sum3 = calMaskfromIntegral(integralImg, x, y + height / 2, width / 2, height / 2);
					int sum4 = calMaskfromIntegral(integralImg, x + width / 2, y + height / 2, width / 2, height / 2);
					result = sum1 - sum2 - sum3 + sum4;
					type = 4;
					//	cout << " x " << x << " y: " << y << " width: " << width << " height: " << height << endl;
					count++;
				}

				//총 feature의 수는 5031개가 나와야됨( local patch에 대해)

				//   cout << endl << "result: " << result << endl;

				//출력확인
				//cout << "result: " << result << endl;
				/*	cout << "x: " << x << endl;
				cout << "y: " << y << endl;
				cout << "type: " << type << endl;*/

				//cout << "count(index):" << count << endl;

				Element element(x, y, width + x, height + y, type, result);
				//vec.push_back(result);

				/*if (1) {
				cout <<"str_x : "<< element.start_x << " str_y : " << element.start_y
				<<" width : " << width<<" height: "<< height
				<<" end_x :"<<element.end_x
				<<" end_y :"<< element.end_y<< " type : " << element.type << "\n"<<endl;
				}*/
				//		cout <<"count"<< count << endl;
				vec.push_back(element);

			}

		}
		//}//

		//}//


		//}//
		//cout << n << endl;

		//for (int i = 0; i < vec.size(); i++) {
		//	//출력확인
		//	cout << i<<" 번째 type: "<<vec.at(i).type << endl;
		//}
		//cout << "size " << vec.size() << endl;



	}

	//count출력
	//	cout <<"feature"<< count << endl;
	//파일 닫기
	fin.close();

	return vec;
}

float calMaskfromIntegral(Mat integral, int x, int y, int w, int h)
{

	//cout << "integral" << endl;
	int endX = x + w >= ORIGINAL_WINDOW_SIZE ? 127 : x + w;
	int endY = y + h >= ORIGINAL_WINDOW_SIZE ? 127 : y + h;
	if (endX >= 128 || endY >= 128) {
		cout << endX << " " << endY << "err";
		int x;
		cin >> x;
	}
	int sum = integral.at<float>(endY, endX);
	sum -= x > 0 ? integral.at<float>(endY, x - 1) : 0;
	sum -= y > 0 ? integral.at<float>(y - 1, endX) : 0;
	sum += x > 0 && y > 0 ? integral.at<float>(y - 1, x - 1) : 0;
	//cout << "sum " <<sum << endl;
	return sum;
}
