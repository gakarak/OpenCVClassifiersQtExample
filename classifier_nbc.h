#ifndef CLASSIFIER_NBC_H
#define CLASSIFIER_NBC_H

#include "classifierinterface.h"

class ClassifierNBC : public ClassifierInterface
{
public:
    ClassifierNBC();
//    ~ClassifierNBC() {}
    //
    void    trainData(const std::vector<cv::Point>& data, const std::vector<int>& labels);
//    int     classify(const cv::Point& p);
    int     classify(int x, int y);
    QString toQString() const;
private:
     cv::NormalBayesClassifier cls;
};

#endif // CLASSIFIER_NBC_H
