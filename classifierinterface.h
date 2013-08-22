#ifndef CLASSIFIERINTERFACE_H
#define CLASSIFIERINTERFACE_H

#include <iostream>
#include <cv.h>
#include <ml.h>
#include <vector>

#include <QString>

enum CLSF_IDX {
    CLSF_IDX_NBC,
    CLSF_IDX_KNN,
    CLSF_IDX_SVM,
    CLSF_IDX_DT,
    CLSF_IDX_BT,
    CLSF_IDX_GBT,
    CLSF_IDX_RF,
    CLSF_IDX_ERT,
    CLSF_IDX_ANN,
    CLSF_IDX_EM,
    CLSF_IDX_EMC,
    //
    CLSF_IDX_NUM
};

class ClassifierInterface
{
public:
    ClassifierInterface();
    virtual ~ClassifierInterface() {}
    //
    virtual void    trainData(const std::vector<cv::Point>& data, const std::vector<int>& labels) = 0;
//    virtual int     classify(const cv::Point& p) = 0;
    virtual int     classify(int x, int y) = 0;
    virtual QString toQString() const = 0;
    virtual int     getMaxClassNum() { return INT_MAX; }
    virtual void    loadAdditionalPoints(std::vector<cv::Point>& extPoints) { extPoints.clear(); } // for support vectors
    //
    bool    isTrained();
    float   calcSelfError();
protected:
    cv::Mat pData;
    cv::Mat lData;
    cv::Mat testSample;
    bool    isTrainedFlag;
    void    loadData(const std::vector<cv::Point>& data, const std::vector<int>& labels);
};

#endif // CLASSIFIERINTERFACE_H
