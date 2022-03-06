
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
#define CLASSIFIERNUM 100

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

//////vector<Element> extractFeature(Mat integralImg);
Mat calculateIntegralFrom(Mat input);
float calMaskfromIntegral(Mat integral, int x, int y, int w, int h);


//CART TREE 속성 생성 관련 전역변수
StrongClassifier strongClassifier;
TrainingData td;
AdaBooster adaboost;

//LBP feature
vector<Element> LBP(Mat src_image/*, CString& resultLbp*/);
 
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
		vector<Element> lbp = LBP(imread(filepath));
		////vector<Element> feature = extractFeature(lbp);
		////

		//posive label==1
		int label = 1;

		trData.push_back(make_pair(lbp, label));

	}

	cout << endl << "positive 완료" << endl;



	//negative 영상 입력 => 특징 추출
	for (int i = 1; i < TR_IMG_PER_CLASS_NEGATIVE + 1; i++)
	{

		sprintf_s(filepath, "%s/%d.bmp", TR_NEG_DIR, i);
		cout << filepath << endl;

		////(진구) : LBP feature 생성으로 변경 필요
		vector<Element> lbp = LBP(imread(filepath));
		////vector<Element> feature = extractFeature(lbp);
		////

		//negative label==-1
		int label = -1;

		trData.push_back(make_pair(lbp, label));

	}

	cout << endl << "negative 완료" << endl;


	///////////////////////////////////////////////
	//feature dimension 확인
	//////for (int i = 0; i < trData.size(); i++) {
	//////int featDim = trData[i].first.size();
	//////int label = trData[i].second;
	//////printf("featDim: %d, label: %d\n", featDim, label);
	//////}


	//Training data 생성

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

		//weakList.at(i).printClassifier();
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

//////vector<Element> extractFeature(Mat integralImg)
//////{
//////
//////	vector<Element> vec;
//////	//
//////	//int type = -1;
//////	//int x, y;
//////	//int width = 2; int height = 2;
//////	//Element element(x, y, width + x, height + y, type, result);
//////	//
//////	//vec.push_back(element);
//////
//////
//////	return vec;
//////}

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

vector<Element> LBP(Mat src_image/*, CString& resultLbp*/)
{
	vector<Element> vec;
	bool affiche = true;
	cv::Mat Image(src_image.rows, src_image.cols, CV_8UC1);
	cv::Mat lbp(src_image.rows, src_image.cols, CV_8UC1);

	if (src_image.channels() == 3)
		cvtColor(src_image, Image, CV_BGR2GRAY);

	unsigned center = 0;
	unsigned center_lbp = 0;

	for (int row = 21; row < Image.rows - 21; row++)

	{

		for (int col = 21; col < Image.cols - 21; col++)

		{

			if (row % 3 == 1 && col % 3 == 1) {

				center = Image.at<uchar>(row, col);
				center_lbp = 0;

				if (center <= Image.at<uchar>(row - 1, col - 1))
					center_lbp += 1;

				if (center <= Image.at<uchar>(row - 1, col))
					center_lbp += 2;

				if (center <= Image.at<uchar>(row - 1, col + 1))
					center_lbp += 4;

				if (center <= Image.at<uchar>(row, col - 1))
					center_lbp += 8;

				if (center <= Image.at<uchar>(row, col + 1))
					center_lbp += 16;

				if (center <= Image.at<uchar>(row + 1, col - 1))
					center_lbp += 32;

				if (center <= Image.at<uchar>(row + 1, col))
					center_lbp += 64;

				if (center <= Image.at<uchar>(row + 1, col + 1))
					center_lbp += 128;

				for (int i = -1; i <= 1; i++) {

					for (int j = -1; j <= 1; j++) {

						lbp.at<uchar>(row + i, col + j) = center_lbp;//저장

			
						//if (center_lbp == 43 ||
						//	center_lbp == 105 ||
						//	center_lbp == 150 ||
						//	center_lbp == 212 ||
						//	center_lbp == 240) {
						//	// 43 105 150 212 240 
						//	circle(src_image, Point(row + i, col + j), 3, Scalar(255, 0, 0), 2);
						//}
					}

				}

				//Lbp feature  저장
				int type = -1;
				int startX = row-1;
				int startY = col-1;
				int width = 2; int height = 2;
				Element element(startX, startY, startX + width, startY+height , type, center_lbp);
				vec.push_back(element);

				//feature 값 확인

				//if (startX <= 30) {
				//	printf("startX: %d , startY : %d , value %d\n", startX, startY, center_lbp);
				//	int p;
				//	cin >> p;
				//}

			}
		}
	}

	//for (int i = 0; i < lbp.size().height; i++) {
	//	for (int j = 0; j < lbp.size().width; j++) {
	//		if (i % 2 == 0 && j % 2 == 0) {
	//			CString paste;//뒤에 이어붙이려고 선언한 스트링
	//			paste.Format("%3d,", lbp.at<uchar>(i, j));//x,y 값을
	//			resultLbp = resultLbp + paste;//이어붙임 이런식으로 (결과1) (결과2) (결과3) ... 붙여짐
	//		}
	//	}
	//}
	//CString paste2;
	//paste2.Format("\n");//요렇게하면
	//resultLbp = resultLbp + paste2;
	//cout << "success" << endl;


	if (affiche == true)
	{
		cv::imshow("image LBP", lbp);
		imshow("result", src_image);
		waitKey(10);
		cv::imshow("grayscale", Image);
		waitKey(10);
	}
	else
	{
		cv::destroyWindow("image LBP");
		cv::destroyWindow("grayscale");
	}

	return vec;
}