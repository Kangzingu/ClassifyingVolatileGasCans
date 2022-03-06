#ifndef _WEAK_CLASSIFIER_H
#define _WEAK_CLASSIFIER_H

#include <string>
#include <vector>

#include "FeatureVector.h"
const int ZERO = 0;

/******************************
 * Class: WeakClassifier
 * ---------------------
 * Weak classifiers have a threshold and a flipped boolean which together tell
 * us if the WeakClassifier correctly classified a feature. It also has a
 * weight.
 */
class WeakClassifier {
	public:
		WeakClassifier(unsigned int in_dim, float in_thresh,bool in_flip, float in_wt,int in_index, float in_classifier_error) :
			dim(in_dim), thresh(in_thresh), flipped(in_flip), am(in_wt),index(in_index), classifier_error(in_classifier_error) {}; // constructor

		int dimension() const;
		float weight() const;
		float threshold() const;
		bool isFlipped() const;

		// Output
		void printClassifier() const;
		void writeClassifier(std::string fname) const;

		//11-10 weak classifier 한개마다 feature를 저장함
		void setFeature(FeatureVector _feature);
		FeatureVector feature;

	private:
		unsigned int dim; // dimension(몇번째 feature인지)
		float thresh; // threshold
		bool flipped;
		float am; // weight

		//11-10
		//Element element;
		//FeatureVector feature;
		
		float classifier_error;

		/////////////////////////////////////////
		int index;//feature에서 나의 위치
	
};

#endif
