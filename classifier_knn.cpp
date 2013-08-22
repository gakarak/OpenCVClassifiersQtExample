#include "classifier_knn.h"

ClassifierKNN::ClassifierKNN()
{
    par_KNN = 1;
}

void ClassifierKNN::trainData(const std::vector<cv::Point> &data, const std::vector<int> &labels)
{
    cls.clear();
    loadData(data, labels);
    cls.train(pData, lData, cv::Mat(), false, par_KNN);
    isTrainedFlag   = true;
}

int ClassifierKNN::classify(int x, int y)
{
    testSample.at<float>(0) = (float)x;
    testSample.at<float>(1) = (float)y;
    return cvRound(cls.find_nearest(testSample, par_KNN));
}

QString ClassifierKNN::toQString() const
{
    return QString("KNN {knn=%1}").arg(par_KNN);
}

void ClassifierKNN::setNumKNN(int knn)
{
    this->par_KNN   = knn;
}
