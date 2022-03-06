#include <iostream>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include <stdlib.h>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define kMeansIter 10

Mat resultPoinnt;
Mat originImage;
const int LOOP_NUM = 10;
const int GOOD_PTS_MAX = 100;
const float GOOD_PORTION = 1.00f;

int64 work_begin = 0;
int64 work_end = 0;

void KMeansClustering(int *x, int *y, int k, int round, float& sse, int* labelResult);
void KMeansClustering(int *x, int *y, int k, int round, float& sse, int* labelResult,
	float centerImageX, float centerImageY, bool isSub, bool isLast,float* centerPointX,float* centerPointY);
float calDistance(float a, float b, float c, float x1, float y1);
int calLBP(float* centerX, float* centerY, int numCenters, Mat sourceImage);
Mat LBP(Mat src_image/*, CString& resultLbp*/);
static void workBegin()
{
	work_begin = getTickCount();
}

static void workEnd()
{
	work_end = getTickCount() - work_begin;
}

static double getTime()
{
	return work_end / ((double)getTickFrequency())* 1000.;
}

struct SURFDetector
{
	Ptr<Feature2D> surf;
	SURFDetector(double hessian = 800.0)
	{
		surf = SURF::create(hessian);
	}
	template<class T>
	void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
	{
		surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
	}
};

template<class KPMatcher>
struct SURFMatcher
{
	KPMatcher matcher;
	template<class T>
	void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
	{
		matcher.match(in1, in2, matches);
	}
};

static Mat drawGoodMatches(
	const Mat& img1,
	const Mat& img2,
	const std::vector<KeyPoint>& keypoints1,
	const std::vector<KeyPoint>& keypoints2,
	std::vector<DMatch>& matches,
	std::vector<Point2f>& scene_corners_
)
{
	//-- Sort matches and preserve top 10% matches
	std::sort(matches.begin(), matches.end());
	std::vector< DMatch > good_matches;
	double minDist = matches.front().distance;
	double maxDist = matches.back().distance;

	//const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
	for (int i = 0; i < /*ptsPairs*/ matches.size(); i++)
	{
		good_matches.push_back(matches[i]);
	}
	std::cout << "\nMax distance: " << maxDist << std::endl;
	std::cout << "Min distance: " << minDist << std::endl;

	//std::cout << "Calculating homography using " << ptsPairs << " point pairs." << std::endl;


	// drawing the results
	Mat img_matches;

	drawMatches(img1, keypoints1, img2, keypoints2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints2[good_matches[i].trainIdx].pt);

		//수정
		//cout << scene.at(i).x<<","<< scene.at(i).y<< endl;
		circle(resultPoinnt, Point(scene.at(i).x, scene.at(i).y), 10, Scalar(255, 0, 0), 4);

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	Mat clusterImage;
	imread("E:/inputImage_grayscale/8.jpg").copyTo(clusterImage);

	int *pointX = new int[scene.size()];
	int *pointY = new int[scene.size()];
	int *labelResult = new int[scene.size()];
	for (int i = 0; i < scene.size(); i++) {
		labelResult[i] = -1;
	}
	Scalar s[12] = {
		Scalar(0,0,255),
		Scalar(0,255,0),
		Scalar(255,0,0),
		Scalar(255,255,0),
		Scalar(255,0,255),
		Scalar(0,255,255),
		Scalar(255,255,255),
		Scalar(127,127,0),
		Scalar(0,127,127),
		Scalar(127,0,127),
		Scalar(127,127,127),
		Scalar(64,64,255)
	};
	for (int i = 0; i < scene.size(); i++) {
		pointX[i] = scene.at(i).x;
		pointY[i] = scene.at(i).y;
	}
	//
	float *sse = new float[kMeansIter];

	/*zingu*/
	float accMin = numeric_limits<float>::min();
	float accMax = 0;
	float* centerTrashPointX = new float[scene.size()];
	float* centerTrashPointY = new float[scene.size()];
	//쓰레기값임 안쓰는거임 걍 넘겨줄라고 만든겅^
	for (int i = 1; i < kMeansIter; i++) {
		KMeansClustering(pointX, pointY, i, 150, sse[i], labelResult, (float)(clusterImage.cols / 2), (float)(clusterImage.rows / 2), false, false, centerTrashPointX, centerTrashPointY);//값은 acc[1]부터 acc[kMeansIter-1]까지 총 19개의 점
		if (sse[i] < accMin)
			accMin = sse[i];
		if (sse[i] > accMax)
			accMax = sse[i];
	}
	for (int i = 1; i < kMeansIter; i++) {
		sse[i] -= accMin;
		sse[i] *= (kMeansIter / ((accMax - accMin)));
	}
	/*zingu*/
	float maxValue = 0;
	int maxIdx = -1;
	float m = (sse[kMeansIter - 1] - sse[1]) / ((kMeansIter - 1) - (1));
	float x1 = kMeansIter - 1; float y1 = sse[kMeansIter - 1];
	float a = m;//(accuracy[kMeansIter - 1] - accuracy[1]) / ((kMeansIter - 1) - (1));
	float b = -1;
	float c = -m * x1 + y1; //-((accuracy[kMeansIter - 1] - accuracy[1]) / ((kMeansIter - 1) - (1))) * (kMeansIter - 1) + accuracy[kMeansIter - 1];

	for (int i = 1; i < kMeansIter; i++) {//총 point의 수 : kMeansIter-1개 (i=1부터라면)
	   //maxValue = max(maxValue, calDistance(a, b, c, i, accuracy[i]));
		if (maxValue < calDistance(a, b, c, i, sse[i])) {
			maxIdx = i;
			maxValue = calDistance(a, b, c, i, sse[i]);
			cout << "changed~!\t" << endl;
		}

		cout << "NOW VALUE: " << calDistance(a, b, c, i, sse[i]) << endl;
		cout << "MAX VALUE: " << maxValue << "\t" << maxIdx << endl;
	}
	cout << maxValue << "\n" << "최종 인식된 사람 수: " << maxIdx << endl;
	//출력이란다
	float optimalAcc = 0;
	float* centerPointX = new float[maxIdx];
	float* centerPointY = new float[maxIdx];
	KMeansClustering(pointX, pointY, maxIdx, 150, optimalAcc, labelResult, (float)(clusterImage.cols / 2), (float)(clusterImage.rows / 2), false, true,centerPointX,centerPointY);//값은 acc[1]부터 acc[kMeansIter-1]까지 총 19개의 점

	for (int i = 0; i < scene.size(); i++) {
		circle(resultPoinnt, Point(scene.at(i).x, scene.at(i).y), 10, s[labelResult[i]], 10);
	}
	printf("%d\n", scene.size());
	int count = 0;
	for (int i = 0; i < scene.size(); i++) {
		if (labelResult[i] != -1) {
			count++;
		}
	}
	printf("labelresultsize: %d, keypointsize: %d\n", count, scene.size());
	cout << "============ " << " 전체 클러스터 내부 점 갯수: " << scene.size() << endl;

	for (int i = 0; i < maxIdx; i++) {
		char fileName[300];
		sprintf_s(fileName, "cluster%d.txt", i);
		//printf("%s\n", fileName);
		ofstream fout(fileName, ios::app);
		//for (int j = 0; j < scene.size(); j++) {
		//	cout << "doing:" << i << "번째 클러스터의 :" << j << endl;
		//	if (labelResult[j] == i) {
				//첫 클러스터의 점들만
		//		fout << pointX[j] << "\t" << pointY[j] << endl;
		//	}
			// 위치를 출력
		//}
		fout << i << "번째, SCALAR:" << s[i] << endl;
		fout.close();
	}
	
	calLBP(centerPointX, centerPointY, maxIdx, originImage);

	///* 클러스터 2단계ㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔ]ㅔㅔㅔㅔㅔㅔㅔㅔㅔㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔ*/
	//for (int subCluster = 0; subCluster < maxIdx; subCluster++) {

	//	vector<int> subVectorX;
	//	vector<int> subVectorY;
	//	int subPointCount = 0;
	//	for (int i = 0; i < scene.size(); i++) {
	//		if (labelResult[i] == subCluster) {
	//			subVectorX.push_back(pointX[i]);
	//			subVectorY.push_back(pointY[i]);
	//			subPointCount++;
	//		}
	//	}
	//	int *subPointX = new int[subPointCount];
	//	int *subPointY = new int[subPointCount];
	//	int *subLabelResult = new int[subPointCount];
	//	for (int i = 0; i < subPointCount; i++) {
	//		subLabelResult[i] = -1;
	//	}

	//	for (int i = 0; i < subPointCount; i++) {
	//		subPointX[i] = subVectorX.at(i);
	//		subPointY[i] = subVectorY.at(i);
	//	}
	//	//
	//	float *subsse = new float[kMeansIter];

	//	/*zingu*/
	//	float subAccMin = numeric_limits<float>::min();
	//	float subAccMax = 0;

	//	//센터 넘겨주기
	//	float subPointCenterX = 0;
	//	float subPointCenterY = 0;
	//	for (int i = 0; i < subPointCount; i++) {
	//		subPointCenterX += subPointX[i];
	//		subPointCenterY += subPointY[i];
	//	}
	//	subPointCenterX /= subPointCount;
	//	subPointCenterY /= subPointCount;
	//	//
	//	for (int i = 1; i < kMeansIter; i++) {
	//		KMeansClustering(subPointX, subPointY, i, 150, subsse[i], subLabelResult, subPointCenterX, subPointCenterY, true);//값은 acc[1]부터 acc[kMeansIter-1]까지 총 19개의 점
	//		if (subsse[i] < subAccMin)
	//			subAccMin = subsse[i];
	//		if (subsse[i] > subAccMax)
	//			subAccMax = subsse[i];
	//	}
	//	for (int i = 1; i < kMeansIter; i++) {
	//		subsse[i] -= subAccMin;
	//		subsse[i] *= (kMeansIter / ((subAccMax - subAccMin)));
	//	}
	//	/*zingu*/
	//	float subMaxValue = 0;
	//	int subMaxIdx = -1;
	//	float subm = (subsse[kMeansIter - 1] - subsse[1]) / ((kMeansIter - 1) - (1));
	//	float subx1 = kMeansIter - 1; float suby1 = subsse[kMeansIter - 1];
	//	float suba = subm;//(accuracy[kMeansIter - 1] - accuracy[1]) / ((kMeansIter - 1) - (1));
	//	float subb = -1;
	//	float subc = -subm * subx1 + suby1; //-((accuracy[kMeansIter - 1] - accuracy[1]) / ((kMeansIter - 1) - (1))) * (kMeansIter - 1) + accuracy[kMeansIter - 1];

	//	for (int i = 1; i < kMeansIter; i++) {//총 point의 수 : kMeansIter-1개 (i=1부터라면)
	//	   //maxValue = max(maxValue, calDistance(a, b, c, i, accuracy[i]));
	//		if (subMaxValue < calDistance(suba, subb, subc, i, subsse[i])) {
	//			subMaxIdx = i;
	//			subMaxValue = calDistance(suba, subb, subc, i, subsse[i]);
	//			cout << "subchanged~!\t" << endl;
	//		}

	//		cout << "subNOW VALUE: " << calDistance(suba, subb, subc, i, subsse[i]) << endl;
	//		cout << "subMAX VALUE: " << subMaxValue << "\t" << subMaxIdx << endl;
	//	}
	//	cout << subMaxValue << "\n" << "최종 인식된sub 사람 수: " << subMaxIdx << endl;
	//	cout << "============ " << subCluster + 1 << " 번클러스터 내부 점 갯수: " << subPointCount << endl;
	//	//출력이란다
	//	float subOptimalAcc = 0;
	//	KMeansClustering(subPointX, subPointY, subMaxIdx, 150, subOptimalAcc, subLabelResult, subPointCenterX, subPointCenterY, true);//값은 acc[1]부터 acc[kMeansIter-1]까지 총 19개의 점

	//	for (int i = 0; i < subPointCount; i++) {
	//		circle(resultPoinnt, Point(subPointX[i], subPointY[i]), 10, s[subLabelResult[i]], 10);
	//	}

	//}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	//binary로 변환
	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0);
	obj_corners[1] = Point(img1.cols, 0);
	obj_corners[2] = Point(img1.cols, img1.rows);
	obj_corners[3] = Point(0, img1.rows);
	std::vector<Point2f> scene_corners(4);

	Mat H = findHomography(obj, scene, RANSAC);
	perspectiveTransform(obj_corners, scene_corners, H);

	scene_corners_ = scene_corners;

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	//line(img_matches,
	//	scene_corners[0] + Point2f((float)img1.cols, 0), scene_corners[1] + Point2f((float)img1.cols, 0),
	//	Scalar(0, 255, 0), 2, LINE_AA);
	//line(img_matches,
	//	scene_corners[1] + Point2f((float)img1.cols, 0), scene_corners[2] + Point2f((float)img1.cols, 0),
	//	Scalar(0, 255, 0), 2, LINE_AA);
	//line(img_matches,
	//	scene_corners[2] + Point2f((float)img1.cols, 0), scene_corners[3] + Point2f((float)img1.cols, 0),
	//	Scalar(0, 255, 0), 2, LINE_AA);
	//line(img_matches,
	//	scene_corners[3] + Point2f((float)img1.cols, 0), scene_corners[0] + Point2f((float)img1.cols, 0),
	//	Scalar(0, 255, 0), 2, LINE_AA);
	return img_matches;
}

//////////////////////////////////////////////////
 /*This program demonstrates the usage of SURF_OCL.
 use cpu findHomography interface to calculate the transformation matrix*/
int main(int argc, char* argv[])
{

	UMat img1, img2;
	Mat binaryImage;
	//imread("logo.jpg").copyTo(img1);
	imread("gray1.jpg").copyTo(img1);
	imread("E:/inputImage_grayscale/8.jpg")
		.copyTo(img2);
	imread("E:/inputImage_grayscale/8.jpg").copyTo(resultPoinnt);
	imread("E:/inputImage_grayscale/8.jpg").copyTo(originImage);
	//threshold(img2, binaryImage, 127,255,CV_THRESH_BINARY);
	binaryImage = Mat::zeros(resultPoinnt.size(), CV_8UC1);


	double surf_time = 0.;

	//declare input/output
	std::vector<KeyPoint> keypoints1, keypoints2;
	std::vector<DMatch> matches;

	UMat _descriptors1, _descriptors2;
	Mat descriptors1 = _descriptors1.getMat(ACCESS_RW),
		descriptors2 = _descriptors2.getMat(ACCESS_RW);

	//instantiate detectors/matchers
	SURFDetector surf;

	SURFMatcher<BFMatcher> matcher;

	//////ofstream out("resultFile.txt", ios::app);
	//-- start of timing section
	for (int i = 0; i <= LOOP_NUM; i++)
	{
		//cout << "cc" << endl;
		if (i == 1) workBegin();
		//cout << "hh" << endl;
		surf(img1.getMat(ACCESS_READ), Mat(), keypoints1, descriptors1);
		surf(img2.getMat(ACCESS_READ), Mat(), keypoints2, descriptors2);
		//cout << "ff" << endl;
		matcher.match(descriptors1, descriptors2, matches);
		printf("%d번째 이미지 종료\n", i + 1);
	}

	////텍스트 파일에 결과 벡터 저장
	//for (int i = 0; i < keypoints2.size(); i++) {
	//	out << keypoints2.at(i).pt.x << "\t" << keypoints2.at(i).pt.y << endl;
	//}
	//out.close();

	/////////////////////////////////
	//int index = 0;
	//ifstream fin("resultFile.txt", ios::app);
	//
	//while (!fin.eof()) {

	//	// 위치를 출력
	//	////fin >> type >> dimension >> start_y >> start_x >> end_y >> end_x >> center_y >> center_x >> height >> width >> size>> value_lbp >> error >> accuracy;

	//	////if (accuracy >= 0.7) {
	//	////	Point point;
	//	////	point.x = center_x;
	//	////	point.y = center_y;
	//	////	circle(resultImage, point, 1, Scalar(0, 0, 255),1);
	//	////}
	//	float x, y;
	//	fin >> x >> y;
	//	cout << x << "," << y << endl;
	//	keypoints2.at(index).pt.x = x;
	//	//keypoints2.at(index).pt.y = y;
	//	index++;

	//	

	//}
	//fin.close();
	///////////////////////////

	workEnd();

	std::cout << "FOUND " << keypoints1.size() << " keypoints on first image" << std::endl;
	std::cout << "FOUND " << keypoints2.size() << " keypoints on second image" << std::endl;

	cout << "finish" << endl;

	for (int i = 0; i < keypoints2.size(); i++) {
		//cout << keypoints2.at(i).pt << endl;
		//circle(resultPoinnt, Point(keypoints2.at(i).pt.x, keypoints2.at(i).pt.y), 2, Scalar(0, 0, 255), 2);
	}
	//binary로변환
	///////////////////////////////////////////////////////////////////////////진구
	//Mat clusterImage;
	//imread("E:/inputImage_grayscale/3.jpg").copyTo(clusterImage);
	//
	//int clusterK = 6;
	//int *pointX = new int[keypoints2.size()];
	//int *pointY = new int[keypoints2.size()];
	//int *labelResult = new int[keypoints2.size()];
	//for (int i = 0; i < keypoints2.size(); i++) {
	//	labelResult[i] = -1;
	//}
	//Scalar s[12] = { 
	//	Scalar(0,0,255),
	//	Scalar(0,255,0),
	//	Scalar(255,0,0),
	//	Scalar(255,255,0),
	//	Scalar(255,0,255),
	//	Scalar(0,255,255),	
	//	Scalar(255,255,255),
	//	Scalar(127,127,0),
	//	Scalar(0,127,127),
	//	Scalar(127,0,127),
	//	Scalar(127,127,127),
	//	Scalar(64,64,255)
	//};
	//for (int i = 0; i < keypoints2.size(); i++) {
	//	pointX[i] = keypoints2.at(i).pt.x;
	//	pointY[i] = keypoints2.at(i).pt.y;
	//}
	////
	//float *sse = new float[kMeansIter];

	///*zingu*/
	//float accMin = numeric_limits<float>::min();
	//float accMax = 0;
	//for (int i = 1; i < kMeansIter; i++) {
	//	KMeansClustering(pointX, pointY, i, 150, sse[i], labelResult, (float)(clusterImage.rows/2), (float)(clusterImage.cols/2),false);//값은 acc[1]부터 acc[kMeansIter-1]까지 총 19개의 점
	//	if (sse[i] < accMin)
	//		accMin = sse[i];
	//	if (sse[i] > accMax)
	//		accMax = sse[i];
	//}
	//for (int i = 1; i < kMeansIter; i++) {
	//	sse[i] -= accMin;
	//	sse[i] *= (kMeansIter / ((accMax - accMin)));
	//}
	///*zingu*/
	//float maxValue = 0;
	//int maxIdx = -1;
	//float m = (sse[kMeansIter - 1] - sse[1]) / ((kMeansIter - 1) - (1));
	//float x1 = kMeansIter - 1; float y1 = sse[kMeansIter - 1];
	//float a = m;//(accuracy[kMeansIter - 1] - accuracy[1]) / ((kMeansIter - 1) - (1));
	//float b = -1;
	//float c = -m * x1 + y1; //-((accuracy[kMeansIter - 1] - accuracy[1]) / ((kMeansIter - 1) - (1))) * (kMeansIter - 1) + accuracy[kMeansIter - 1];

	//for (int i = 1; i < kMeansIter; i++) {//총 point의 수 : kMeansIter-1개 (i=1부터라면)
	//   //maxValue = max(maxValue, calDistance(a, b, c, i, accuracy[i]));
	//	if (maxValue < calDistance(a, b, c, i, sse[i])) {
	//		maxIdx = i;
	//		maxValue = calDistance(a, b, c, i, sse[i]);
	//		cout << "changed~!\t" << endl;
	//	}

	//	cout << "NOW VALUE: " << calDistance(a, b, c, i, sse[i]) << endl;
	//	cout << "MAX VALUE: " << maxValue << "\t" << maxIdx << endl;
	//}
	//cout << maxValue << "\n" << "최종 인식된 사람 수: " << maxIdx << endl;
	////출력이란다
	//float optimalAcc = 0;
	//KMeansClustering(pointX, pointY, maxIdx, 150, optimalAcc, labelResult, (float)(clusterImage.rows / 2), (float)(clusterImage.cols / 2), false);//값은 acc[1]부터 acc[kMeansIter-1]까지 총 19개의 점

	//for (int i = 0; i < keypoints2.size(); i++) {
	//	circle(resultPoinnt, Point(keypoints2.at(i).pt.x, keypoints2.at(i).pt.y),10, s[labelResult[i]], 10);
	//}
	//printf("%d\n", keypoints2.size());
	//int count = 0;
	//for (int i = 0; i < keypoints2.size(); i++) {
	//	if (labelResult[i] != -1) {
	//		count++;
	//	}
	//}
	//printf("labelresultsize: %d, keypointsize: %d\n", count, keypoints2.size());
	//cout << "============ " << " 전체 클러스터 내부 점 갯수: " << keypoints2.size() << endl;
	//
	//ofstream fout("cluster.txt", ios::app);

	//for (int i = 0; i < keypoints2.size(); i++) {
	//	if (labelResult[i] == 0) {
	//		//첫 클러스터의 점들만
	//		fout << pointX[i] << "\t" << pointY[i]<<endl;
	//	}
	//	// 위치를 출력
	//}
	//fout.close();

	///* 클러스터 2단계ㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔ]ㅔㅔㅔㅔㅔㅔㅔㅔㅔㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔㅔ*/
	//for (int subCluster= 0; subCluster < maxIdx; subCluster++) {

	//	vector<int> subVectorX;
	//	vector<int> subVectorY;
	//	int subPointCount = 0;
	//	for (int i = 0; i < keypoints2.size(); i++) {
	//		if (labelResult[i] == subCluster) {
	//			subVectorX.push_back(pointX[i]);
	//			subVectorY.push_back(pointY[i]);
	//			subPointCount++;
	//		}
	//	}
	//	int *subPointX = new int[subPointCount];
	//	int *subPointY = new int[subPointCount];
	//	int *subLabelResult = new int[subPointCount];
	//	for (int i = 0; i < subPointCount; i++) {
	//		subLabelResult[i] = -1;
	//	}
	//	
	//	for (int i = 0; i < subPointCount; i++) {
	//		subPointX[i] = subVectorX.at(i);
	//		subPointY[i] = subVectorY.at(i);
	//	}
	//	//
	//	float *subsse = new float[kMeansIter];

	//	/*zingu*/
	//	float subAccMin = numeric_limits<float>::min();
	//	float subAccMax = 0;

	//	//센터 넘겨주기
	//	float subPointCenterX = 0;
	//	float subPointCenterY = 0;
	//	for (int i = 0; i < subPointCount; i++) {
	//		subPointCenterX += subPointX[i];
	//		subPointCenterY += subPointY[i];
	//	}
	//	subPointCenterX /= subPointCount;
	//	subPointCenterY /= subPointCount;
	//	//
	//	for (int i = 1; i < kMeansIter; i++) {
	//		KMeansClustering(subPointX, subPointY, i, 150, subsse[i], subLabelResult, subPointCenterX, subPointCenterY,true);//값은 acc[1]부터 acc[kMeansIter-1]까지 총 19개의 점
	//		if (subsse[i] < subAccMin)
	//			subAccMin = subsse[i];
	//		if (subsse[i] > subAccMax)
	//			subAccMax = subsse[i];
	//	}
	//	for (int i = 1; i < kMeansIter; i++) {
	//		subsse[i] -= subAccMin;
	//		subsse[i] *= (kMeansIter / ((subAccMax - subAccMin)));
	//	}
	//	/*zingu*/
	//	float subMaxValue = 0;
	//	int subMaxIdx = -1;
	//	float subm = (subsse[kMeansIter - 1] - subsse[1]) / ((kMeansIter - 1) - (1));
	//	float subx1 = kMeansIter - 1; float suby1 = subsse[kMeansIter - 1];
	//	float suba = subm;//(accuracy[kMeansIter - 1] - accuracy[1]) / ((kMeansIter - 1) - (1));
	//	float subb = -1;
	//	float subc = -subm * subx1 + suby1; //-((accuracy[kMeansIter - 1] - accuracy[1]) / ((kMeansIter - 1) - (1))) * (kMeansIter - 1) + accuracy[kMeansIter - 1];

	//	for (int i = 1; i < kMeansIter; i++) {//총 point의 수 : kMeansIter-1개 (i=1부터라면)
	//	   //maxValue = max(maxValue, calDistance(a, b, c, i, accuracy[i]));
	//		if (subMaxValue < calDistance(suba, subb, subc, i, subsse[i])) {
	//			subMaxIdx = i;
	//			subMaxValue = calDistance(suba, subb, subc, i, subsse[i]);
	//			cout << "subchanged~!\t" << endl;
	//		}

	//		cout << "subNOW VALUE: " << calDistance(suba, subb, subc, i, subsse[i]) << endl;
	//		cout << "subMAX VALUE: " << subMaxValue << "\t" << subMaxIdx << endl;
	//	}
	//	cout << subMaxValue << "\n" << "최종 인식된sub 사람 수: " << subMaxIdx << endl;
	//	cout <<"============ "<< subCluster+1<<" 번클러스터 내부 점 갯수: " << subPointCount << endl;
	//	//출력이란다
	//	float subOptimalAcc = 0;
	//	KMeansClustering(subPointX, subPointY, subMaxIdx, 150, subOptimalAcc, subLabelResult, subPointCenterX, subPointCenterY, true);//값은 acc[1]부터 acc[kMeansIter-1]까지 총 19개의 점

	//	for (int i = 0; i < subPointCount; i++) {
	//		circle(resultPoinnt, Point(subPointX[i],subPointY[i]), 10, s[subLabelResult[i]], 10);
	//	}
	//	
	//}
	/* 클러스터 2단계 끝ㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌㅌ*/

	//int clusterK = 6;
	//int *pointX = new int[keypoints2.size()];
	//int *pointY = new int[keypoints2.size()];
	//int *labelResult = new int[keypoints2.size()];
	//for (int i = 0; i < keypoints2.size(); i++) {
	//	labelResult[i] = -1;
	//}
	//int clusterColor[7] = { 255,210,180,135,90,45,0 };
	//float sse = 0;
	//Scalar s[7] = { Scalar(0,0,255),Scalar(255,255,0), Scalar(255,0,255), Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,0), Scalar(0,255,255) };
	//for (int i = 0; i < keypoints2.size(); i++) {
	//	pointX[i] = keypoints2.at(i).pt.x;
	//	pointY[i] = keypoints2.at(i).pt.y;
	//}
	//KMeansClustering(pointX, pointY, clusterK, 150, sse, labelResult,binaryImage);
	////label 범위 0 ~ k-1
	//for (int i = 0; i < keypoints2.size(); i++) {
	//	circle(resultPoinnt, Point(keypoints2.at(i).pt.x, keypoints2.at(i).pt.y), 5, s[labelResult[i]], 5);
	//}
	//printf("%d", keypoints2.size());
	//int count = 0;
	//for (int i = 0; i < keypoints2.size(); i++) {
	//	if (labelResult[i] != -1) {
	//		count++;
	//	}
	//}
	//printf("labelresultsize: %d, keypointsize: %d", count, keypoints2.size());
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*for (int i = 0; i < binaryImage.rows; i++) {
		for (int j = 0; j < binaryImage.cols; j++) {
			if (binaryImage.at<uchar>(i, j) == 0) {
				binaryImage.at<uchar>(i, j) = 255;
			}
		}
	}
*/
	int rectSize = 3;
	for (int i = 0; i < keypoints2.size(); i++) {
		rectangle(binaryImage, Rect(
			Point(keypoints2.at(i).pt.x - rectSize, keypoints2.at(i).pt.y - rectSize),
			Point(keypoints2.at(i).pt.x + rectSize, keypoints2.at(i).pt.y + rectSize)),
			Scalar(255), 1);

		//binaryImage.at<uchar>(keypoints2.at(i).pt.x, keypoints2.at(i).pt.y) = 255;
	}
	//for (int i = 0; i < grayZingu.rows; i++) {
	//	for (int j = 0; j < grayZingu.cols; j++) {
	//		if (grayZingu.at<uchar>(i, j) != 0) {
	//			binaryImage.at<uchar>(i, j) = 255;
	//			//circle(resultPoinnt, Point(i, j), 2, Scalar(255), 2);
	//		}
	//	}
	//}
/*
	for (int i = 0; i < binaryImage.rows; i++) {
		for (int j = 0; j < binaryImage.cols; j++) {
			if (binaryImage.at<uchar>(i, j) != 0) {
				binaryImage.at<uchar>(i, j) = 255.0;
			}
		}
	}*/
	std::vector<Point2f> corner;
	Mat img_matches = drawGoodMatches(img1.getMat(ACCESS_READ), img2.getMat(ACCESS_READ), keypoints1, keypoints2, matches, corner);

	//-- Show detected matches

	//namedWindow("surf matches", 0);
	//imshow("surf matches", img_matches);
	////////////imshow("binaryImage", binaryImage);
	imwrite("binaryImage.bmp", binaryImage);

	imwrite("result1.bmp", resultPoinnt);
	waitKey(0);

	return EXIT_SUCCESS;
}
void KMeansClustering(int *x, int *y, int k, int round, float& sse, int* labelResult)
{
	srand((unsigned int)time(NULL));
	sse = 0;
	float *centerX = new float[k];
	float *centerY = new float[k];
	float *distance = new float[k];
	int numPoints = _msize(x) * 2 / sizeof(x);
	printf("cmsize:%d, sizeof(x):%d\n", _msize(x), sizeof(x));
	int *label = new int[numPoints];
	int *labelCountSum = new int[k];

	for (int i = 0; i < k; i++) {
		labelCountSum[i] = 0;
		distance[i] = 0;
		bool isRepeated = false;
		int randIndex = rand() % numPoints;
		for (int j = 0; j < i; j++) {
			if (centerX[j] == x[randIndex] &&
				centerY[j] == y[randIndex]) {
				isRepeated = true;
				break;
			}
		}
		if (isRepeated == true) {
			i--;
			continue;
		}
		centerX[i] = x[randIndex];
		centerY[i] = y[randIndex];
	}
	//랜덤으로 k개 군집 지정
	for (int end = 0; end < round; end++) {
		for (int i = 0; i < numPoints; i++) {
			int min = numeric_limits<int>::max();
			int minIndex;
			for (int j = 0; j < k; j++) {
				distance[j] = sqrt(pow(x[i] - centerX[j], 2) + pow(y[i] - centerY[j], 2));
				//j번째 군집의 중심과 i번째 군집의 거리를 구하겠당
				if (distance[j] < min) {
					min = distance[j];
					minIndex = j;
				}
				//최소 거리 군집 인덱스 구한당
			}
			label[i] = minIndex;
			//최소 거리 군집에 할당
			//군집이 바꼈으니 센터도 바뀔거얌ㅎ      
		}

		for (int i = 0; i < k; i++) {
			centerX[i] = 0;
			centerY[i] = 0;
			labelCountSum[i] = 0;
		}
		for (int i = 0; i < numPoints; i++) {
			centerX[label[i]] += x[i];
			centerY[label[i]] += y[i];
			labelCountSum[label[i]]++;
		}
		for (int i = 0; i < k; i++) {
			centerX[i] /= labelCountSum[i];
			centerY[i] /= labelCountSum[i];

		}
	}
	for (int i = 0; i < numPoints; i++) {
		sse += sqrt(pow(x[i] - centerX[label[i]], 2) + pow(y[i] - centerY[label[i]], 2));
		labelResult[i] = label[i];
	}

	printf("%d\n", numPoints);
	/*CString str;
	str.Format("%f", sse);
	AfxMessageBox(str);*/
	cout << sse << " : 군집수(k) = " << k << " < SSE value " << endl;
	//if (k == 4) {
	//	for (int i = 0; i < k; i++)
	//		cvCircle(pImg, CvPoint(centerX[i], centerY[i]), 5, CV_RGB(0, 0, 255), 3, 8, 0);
	//}//sizeof(centerX);

	delete[] centerX;
	delete[] centerY;
	delete[] distance;
	delete[] label;
	delete[] labelCountSum;
}

int imageCount = 0;
void KMeansClustering(int *x, int *y, int k, int round, float& sse, int* labelResult, float centerImageX, float centerImageY, bool isSub, bool isLast, float* centerPointX, float* centerPointY)
{
	//이미지 상 중점
	int flag = 0;
	//초기 K 영역별 지정 플래그값

	srand((unsigned int)time(NULL));
	sse = 0;
	float *centerX = new float[k];
	float *centerY = new float[k];
	float *distance = new float[k];
	int numPoints = _msize(x) * 2 / sizeof(x);
	printf("cmsize:%d, sizeof(x):%d\n", _msize(x), sizeof(x));
	int *label = new int[numPoints];
	int *labelCountSum = new int[k];

	for (int i = 0; i < k; i++) {
		labelCountSum[i] = 0;
		distance[i] = 0;
		bool isRepeated = false;
		int randIndex = rand() % numPoints;
		for (int j = 0; j < i; j++) {
			if (centerX[j] == x[randIndex] &&
				centerY[j] == y[randIndex]) {
				isRepeated = true;
				break;
			}
		}
		switch (flag)
		{
		case 0://왼위
			if (x[randIndex] < centerImageX&&
				y[randIndex] < centerImageY) {
				//통과
				flag++;
			}
			else {
				isRepeated = true;
			}
			break;
		case 1://오른위
			if (x[randIndex] < centerImageX&&
				y[randIndex] >= centerImageY) {
				//통과
				flag++;
			}
			else {
				isRepeated = true;
			}
			break;
		case 2://왼아래
			if (x[randIndex] >= centerImageX &&
				y[randIndex] < centerImageY) {
				//통과
				flag++;
			}
			else {
				isRepeated = true;
			}
			break;
		case 3://오른아래
			if (x[randIndex] >= centerImageX &&
				y[randIndex] >= centerImageY) {
				//통과
				flag = 0;
			}
			else {
				isRepeated = true;
			}
			break;
		default:
			break;
		}
		if (isRepeated == true) {
			i--;
			continue;
		}
		centerX[i] = x[randIndex];
		centerY[i] = y[randIndex];
		char subfile[500];
		if (isSub) {
			printf("flag:%d  centerx[i]:%f   centerY[i]:%f  ceterImageX:%f  centerImageY:%f\n", flag, centerX[i], centerY[i],
				centerImageX, centerImageY);
			circle(resultPoinnt, Point(centerX[i], centerY[i]), 40, Scalar(0, 0, 0), 40);
			sprintf_s(subfile, "centerImage%d.jpg", imageCount);
			imageCount++;
			imwrite(subfile, resultPoinnt);
			//imshow("centerImage", resultPoinnt);

		}

	}
	//랜덤으로 k개 군집 지정
	for (int end = 0; end < round; end++) {
		for (int i = 0; i < numPoints; i++) {
			int min = numeric_limits<int>::max();
			int minIndex;
			for (int j = 0; j < k; j++) {
				distance[j] = sqrt(pow(x[i] - centerX[j], 2) + pow(y[i] - centerY[j], 2));
				//j번째 군집의 중심과 i번째 군집의 거리를 구하겠당
				if (distance[j] < min) {
					min = distance[j];
					minIndex = j;
				}
				//최소 거리 군집 인덱스 구한당
			}
			label[i] = minIndex;
			//최소 거리 군집에 할당
			//군집이 바꼈으니 센터도 바뀔거얌ㅎ      
		}

		for (int i = 0; i < k; i++) {
			centerX[i] = 0;
			centerY[i] = 0;
			labelCountSum[i] = 0;
		}
		for (int i = 0; i < numPoints; i++) {
			centerX[label[i]] += x[i];
			centerY[label[i]] += y[i];
			labelCountSum[label[i]]++;
		}
		for (int i = 0; i < k; i++) {
			centerX[i] /= labelCountSum[i];
			centerY[i] /= labelCountSum[i];

		}
	}
	for (int i = 0; i < numPoints; i++) {
		sse += sqrt(pow(x[i] - centerX[label[i]], 2) + pow(y[i] - centerY[label[i]], 2));
		labelResult[i] = label[i];
	}
	//최종 클러스터 센터 네모쳐서 출력
	if (isLast == true) {
		int rectSize = 200;
		for (int i = 0; i < k; i++) {
			rectangle(resultPoinnt,
				Rect(
					Point(centerX[i] - rectSize, centerY[i] - rectSize),
					Point(centerX[i] + rectSize, centerY[i] + rectSize)),
				Scalar(0, 0, 0), 5);
			centerPointX[i] = centerX[i];
			centerPointY[i] = centerY[i];
		}
	}

	//printf("%d\n", numPoints);
	/*CString str;
	str.Format("%f", sse);
	AfxMessageBox(str);*/
	cout << sse << " : 군집수(k) = " << k << " < SSE value " << endl;
	//if (k == 4) {
	//	for (int i = 0; i < k; i++)
	//		cvCircle(pImg, CvPoint(centerX[i], centerY[i]), 5, CV_RGB(0, 0, 255), 3, 8, 0);
	//}//sizeof(centerX);

	delete[] centerX;
	delete[] centerY;
	delete[] distance;
	delete[] label;
	delete[] labelCountSum;
}
float calDistance(float a, float b, float c, float x1, float y1) {

	float numerator = abs(a*x1 + b * y1 + c);
	float denominator = sqrt(a*a + b * b);
	return (numerator / denominator);
}


#define INPUT_IMAGE "E:/inputImage_grayscale"
int calLBP(float* centerX, float* centerY, int numCenters, Mat sourceImage) {
	Mat resultImage;
	cout << "들어옴^^" << endl;
	(LBP(sourceImage)).copyTo(resultImage);

	cout << "LBP 끝남^^" << endl;
	cout << resultImage.rows << "/" << resultImage.cols << "resultImage" << endl;
	cout << numCenters << "nCenters" << endl;

	ofstream fout("outputLBP.txt", ios::app);
	int outputCopy;
	//for (int i = 0; i < resultImage.rows; i++) {
	//	for (int j = 0; j < resultImage.cols; j++) {
	//		outputCopy = resultImage.at<uchar>(i, j);
	//		fout << outputCopy << "\t";
	//	}
	//	fout << endl;
		// 위치를 출력
	//}
	fout.close();
	/*for (int i = 0; i < numCenters; i++) {
		for (int j = -64; j < 63; j++) {
			for (int k = -64; k < 63; k++) {
				cout << resultImage.at<uchar>((int)(centerX[i]+j), (int)(centerY[i]+k));
			}
			cout << endl;
		}
	}*/
	return 1;
}
Mat LBP(Mat src_image/*, CString& resultLbp*/)
{
	//vector<Element> vec;
	bool affiche = true;
	cv::Mat Image(src_image.rows, src_image.cols, CV_8UC1);
	cv::Mat lbp(src_image.rows, src_image.cols, CV_8UC1);

	if (src_image.channels() == 3)
		cvtColor(src_image, Image, CV_BGR2GRAY);

	unsigned center = 0;
	unsigned center_lbp = 0;

	for (int row = 1; row < Image.rows - 1; row++)

	{

		for (int col = 1; col < Image.cols - 1; col++)

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
					}

				}
				//int psp = lbp.at<uchar>(row, col);
				//cout << psp << endl;

			/*	if (row % 100 == 1)
					cout << center_lbp;
*/
				//Lbp feature  저장
				//int type = -1;
				//int startX = row - 1;
				//int startY = col - 1;
				//int width = 2; int height = 2;
				//Element element(startX, startY, startX + width, startY + height, type, center_lbp);
				//vec.push_back(element);

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
		//cv::imshow("image LBP", lbp);
		//imshow("result", src_image);
		waitKey(10);
		//cv::imshow("grayscale", Image);
		waitKey(10);
	}
	else
	{
		cv::destroyWindow("image LBP");
		cv::destroyWindow("grayscale");
	}

	return lbp;
}


