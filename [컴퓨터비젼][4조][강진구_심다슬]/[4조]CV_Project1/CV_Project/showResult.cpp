
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
	//sprintf_s(filepath, "K:/inputImage/input.bmp");
	sprintf_s(filepath, "160.bmp");
	//sprintf_s(filepath, "K:/positive_grayscale/1.bmp");
	//sprintf_s(filepath, "input4.bmp");
	mainImage =imread(filepath);

	// 관심영역 자르기 // (set ROI (X, Y, W, H)).
	//resize(mainImage, mainImage, Size(64, 64), 0, 0, CV_INTER_LINEAR);
	//resize(mainImage, mainImage, Size(1024, 1024), 0, 0, CV_INTER_LINEAR);
	//imshow("resize.image", mainImage);
	//printf("main image size:%d\n", mainImage.size());
	//////
	//Rect rect(400, 400, 128, 128);//col,row
	//////printf("rec size:%d\n", rect.size());
	//mainImage = mainImage(rect);
	//////// show
	//imshow("crop.image", mainImage);


	//파일 읽기
	ifstream fin("weakClassifier_info.txt");

	if (!fin)
	{
		cerr << "[weakClassifier_info read fail]" << endl;
		return -1;
	}

	float dimension, error, accuracy, value_lbp=-1;
	int start_x, start_y, end_x, end_y, type, center_y, center_x, height, width, size;
	
	Mat resultImage = mainImage.clone();


	while (!fin.eof()) {

		// 위치를 출력
		fin >> type >> dimension >> start_y >> start_x >> end_y >> end_x >> center_y >> center_x >> height >> width >> size>> value_lbp >> error >> accuracy;

		if (accuracy >= 0.7) {
			Point point;
			point.x = center_x;
			point.y = center_y;
			circle(resultImage, point, 1, Scalar(0, 0, 255),1);
		}

		//값을 출력
		//fin >> type >> dimension >> start_y >> start_x >> end_y >> end_x >> center_y >> center_x >> height >> width >> size >> value_lbp>>error >> accuracy;
		if (accuracy >= 0.7) {
			/*if (value_lbp >= 100 && value_lbp <= 150) {*/
				//valList.push_back(value_lbp);
		/*	}*/
			
		}
	}
	fin.close();

	//weak classifier 들의 value를 텍스트 파일로 출력함
	//////ofstream out("weakValueList.txt", ios::app);
	//////for (int i = 0; i < valList.size(); i++) {
	//////	out << valList.at(i) << endl;
	//////}

	//weaklIST 들의 value 출력함
	//////for (int i = 0; i < valList.size(); i++) {
	//////	cout << valList.at(i) << endl;
	//////}

	LBP(resultImage);
	printf("pointCount: %d\n", pointCount);

	/////////////////////////////////////////////////////////////

	
	imwrite("160_result.bmp", resultImage);
	imshow("resultImage", resultImage);

	////////out.close();

	waitKey();

	
		
	
	return 0;
}

Mat LBP(Mat src_image)
{
	ofstream outFile("LBPfeature.txt", ios::app);
	bool affiche = true;
	cv::Mat Image(src_image.rows, src_image.cols, CV_8UC1);
	cv::Mat lbp(src_image.rows, src_image.cols, CV_8UC1);

	//요기까지
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

						lbp.at<uchar>(row + i, col + j) = center_lbp;



					}

				}
				
				outFile << center_lbp<<"\t"<<row<< "\t" <<col << endl;

				for (int k = 0; k < valList.size(); k++) {
					if (center_lbp == valList.at(k)) {
						pointCount++;
						if (pointCount % 100000 == 0) {
							cout << pointCount << endl;
						}
						circle(src_image, Point(row, col), 1, Scalar(0, 0, 255), 1);
					}
				}

			}
		}
	}
	
	if (affiche == true)
	{
		cv::imshow("image LBP", lbp);
		imshow("result", src_image);
		waitKey(10);
		//cv::imshow("grayscale", Image);
		waitKey(10);
	}
	else
	{
		cv::destroyWindow("image LBP");
		//cv::destroyWindow("grayscale");
	}
	outFile.close();
	return lbp;
}