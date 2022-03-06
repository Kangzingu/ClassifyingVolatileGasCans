
#include <cstdio>
#include "FeatureVector.h"
#include "Element.h"
#include <iostream>
#include <fstream>
using namespace std;

//constructor
FeatureVector::FeatureVector(const vector<Element> &in_vec, 
		int in_val) : mlabel(in_val), wt(-1){
	
	// copy vector of floats
	for (int i=0; i<in_vec.size(); i++)
		fvec.push_back(in_vec[i]);
}

//FeatureVector::FeatureVector(const std::vector<float> &in_vec,
//	int in_val) : mlabel(in_val), wt(-1) {
//	
//	// copy vector of floats
//	for (int i = 0; i<in_vec.size(); i++)
//		this->ffvec.push_back(in_vec[i]);
//
//}

FeatureVector::FeatureVector()
{
}

FeatureVector::FeatureVector(const FeatureVector &other) {
  this->fvec = other.fvec;
  this->mlabel = other.mlabel;
  this->wt = other.wt;

}

// returns size of feature vector (useful in addFeature in TrainingData)
unsigned int FeatureVector::size() const { return fvec.size(); }

// more getter methods
int FeatureVector::label() const { return mlabel; }
float FeatureVector::weight() const { return wt; }
float FeatureVector::at(unsigned int i) const { return fvec[i].value; }

// sets weight to given value 
void FeatureVector::setWeight(float weight) { wt = weight; }

// prints out all instance fields of a feature
void FeatureVector::printFeature() const {
	for ( int i = 0; i < fvec.size(); i++) {
		printf("[%d]: %.3f,type: %d, start : (%d,%d), end : (%d,%d)", i, fvec[i].value, fvec[i].type,fvec[i].start_x,fvec[i].start_y,
			fvec[i].end_x,fvec[i].end_y
			);
	}
		
	printf("\n\tval: %d, weight: %f\n",mlabel,wt);
}

void FeatureVector::printFeature(int index) const
{
	printf("[%d]: %.3f, start : (%d,%d), end : (%d,%d), type: %d ", index, fvec[index].value, fvec[index].start_x, fvec[index].start_y,
		fvec[index].end_x, fvec[index].end_y,
		fvec[index].type);

}
void FeatureVector::writeFeature(string fname,int index) const {
	
	ofstream fout(fname,ios::app);

	fout 
		<< "\t" << fvec[index].type << "\t"
		<< index << "\t"
		
		<< fvec[index].start_x
		<< "\t" << fvec[index].start_y
		<< "\t" << fvec[index].end_x
		<< "\t" << fvec[index].end_y
		<< "\t" << (fvec[index].start_y + fvec[index].end_y) / 2
		<< "\t" << (fvec[index].start_x + fvec[index].end_x) / 2
		<<"\t"<< fvec[index].end_y- fvec[index].start_y 
		<< "\t" << fvec[index].end_x - fvec[index].start_x
		<<"\t"<< (fvec[index].end_y - fvec[index].start_y)*(fvec[index].end_x - fvec[index].start_x)
		<< "\t"<<fvec[index].value
		<< "\t";

	fout.close();                



}
vector<Element> FeatureVector::getFvec()
{
	return this->fvec;
}

