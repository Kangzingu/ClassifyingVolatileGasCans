#include <fstream>
#include <iostream>
#include "WeakClassifier.h"

using namespace std;

// getter methods
int WeakClassifier::dimension() const { return dim; }
float WeakClassifier::weight() const { return am; }
float WeakClassifier::threshold() const { return thresh; }
bool WeakClassifier::isFlipped() const { return flipped; }

// prints all instance fields of a weak classifier
void WeakClassifier::printClassifier() const {
	printf("error: %f\t index: %d\t dimension: %d\t threshold: %.10f\t flipped: %s\t am: %f\n",
			classifier_error,index,dim, thresh, (flipped)?"true ":"false", am);
	
	//element Ãâ·Â

	printf("element: ");
	feature.printFeature(dim);
	cout << endl;

}

// writes classifier to file
void WeakClassifier::writeClassifier(string fname) const {

	feature.writeFeature(fname.c_str(), dim);
	ofstream outFile(fname.c_str(),ios::app);
	outFile << classifier_error << "\t" << 1- classifier_error << endl;

	outFile.close();
}


void WeakClassifier::setFeature(FeatureVector _feature)
{
	this->feature = _feature;
}
