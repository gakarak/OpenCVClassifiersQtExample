#ifndef CLASSIFIER_KNN_H
#define CLASSIFIER_KNN_H

#include "classifierinterface.h"

class ClassifierKNN : public ClassifierInterface
{
public:
    ClassifierKNN();
    void    trainData(const std::vector<cv::Point>& data, const std::vector<int>& labels);
//    int     classify(const cv::Point& p);
    int     classify(int x, int y);
    QString toQString() const;
    //
    void setNumKNN(int knn);
private:
     cv::KNearest cls;
     int par_KNN;
};

#endif // CLASSIFIER_KNN_H
