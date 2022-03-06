
#ifndef _FEATURE_VECTOR_H
#define _FEATURE_VECTOR_H
#pragma once
#include "Element.h"
#include<vector>
using namespace std;
static const int POS = 1;
static const int NEG = -1;

/**********************************
 * Class: FeatureVector
 * --------------------
 * Feature vectors have a value, weight, and vector of features (floats) that
 * can be compared with other feature vectors
 */
class FeatureVector {
	public:
		FeatureVector(const std::vector<Element> &in_vec, 
				int in_val); // constructor

		//FeatureVector(const std::vector<float> &in_vec,
		//	int in_val); // constructor

		FeatureVector();
		FeatureVector(const FeatureVector &other);

		unsigned int size() const;
		int label() const;
		float weight() const;
		float at(unsigned int i) const;

		void setWeight(float weight);

		void printFeature() const;
		void FeatureVector::writeFeature(string fname,int index) const;
		//11-10 elemtent 하나만 출력
		void printFeature(int position)const;

		vector<Element> getFvec();
		//vector<float> ffvec;//feature안의 원소가  float 
	private:
		//std::vector <float> fvec; // feature vector
		int mlabel; // label (POS or NEG)
		vector<Element> fvec;// feature안의 원소가 int가 아닌 Element type임


		
		float wt; // weight
};

#endif
