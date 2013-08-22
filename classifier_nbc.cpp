#include "classifier_nbc.h"

ClassifierNBC::ClassifierNBC()
{
}

void ClassifierNBC::trainData(const std::vector<cv::Point> &data, const std::vector<int> &labels)
{
    cls.clear();
    loadData(data, labels);
//    std::cout << "data: cols=" << pData.cols << ", rows=" << pData.rows << " : " << pData << std::endl;
//    std::cout << "labl: cols=" << lData.cols << ", rows=" << lData.rows << " : " << lData << std::endl;
//    cls = cv::NormalBayesClassifier(pData, lData);
    cls.train(pData, lData);
    isTrainedFlag   = true;
}

/*
int ClassifierNBC::classify(const cv::Point &p)
{
    return classify(p.x, p.y);
}
*/


int ClassifierNBC::classify(int x, int y)
{
    testSample.at<float>(0) = (float)x;
    testSample.at<float>(1) = (float)y;
//    std::cout << "cols=" << testSample.cols << ", rows=" << testSample.rows << " : " << testSample << std::endl;
    return cvRound(cls.predict(testSample));
}

QString ClassifierNBC::toQString() const
{
    return QString("NBC");
}
