/////////////////////////////////////////////////////gray convert
//
//#pragma once
//// etc header
//#include <iostream>
//#include <fstream>
//#include<stdio.h>
//#include <string>
//
//
////opencv header
//#include "opencv2/opencv.hpp"
//
//using namespace std;
//using namespace cv;
//
////CLASSIFIER ��
//#define CLASSIFIERNUM 20
//
////�̹��� ����
//#define WIDTH   128
//#define HEIGHT  128
//#define   ORIGINAL_WINDOW_SIZE   128//width, height ����
//
////�н� ������ ��
//#define   TR_IMG_PER_CLASS_POSITIVE 60
//#define   TR_IMG_PER_CLASS_NEGATIVE 60
//
////�н� ������ ���
//#define TR_POS_DIR "K:/positive"
//#define TR_NEG_DIR "K:/negative"
//
//// ��� �̹��� ���� ���
//#define RESULT_IMAGE "K:/resultImage"
//
//#define INPUT_IMAGE "E:/inputImage"
//#define GRAY_IMAGE "E:/inputImage_grayscale"
//
////ó�� ��� ����(�ٽ� �α�)
//#define   DONE 1
//#define   FAIL 0
//
//
//int pointCount = 0;
//
//int main() {
//
//	Mat mainImage;
//
//	char filepath[300];
//	char resultpath[300];
//
//	for (int i = 1; i < 8; i++) {
//		sprintf_s(filepath, "%s/%d.jpg", INPUT_IMAGE, i);
//		mainImage = imread(filepath);
//
//		cvtColor(mainImage, mainImage, CV_RGB2GRAY);
//		sprintf_s(resultpath, "%s/%d.jpg", GRAY_IMAGE, i);
//
//		imwrite(resultpath, mainImage);
//	}
//	
//	
//
//	
//	
//	//imwrite("templet3_gray.jpg", mainImage);
//
//	return 0;
//}
