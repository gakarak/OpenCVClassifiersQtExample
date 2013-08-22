#include "classifierinterface.h"

ClassifierInterface::ClassifierInterface() {
    testSample      = cv::Mat::zeros(1,2,CV_32FC1);
    isTrainedFlag   = false;
}

void ClassifierInterface::loadData(const std::vector<cv::Point> &data, const std::vector<int> &labels)
{
    cv::Mat(data).copyTo(pData);
    cv::Mat(labels).copyTo(lData);
    pData   = pData.reshape(1, pData.rows);
    pData.convertTo(pData, CV_32FC1);
}

bool ClassifierInterface::isTrained()
{
    return isTrainedFlag;
}

float ClassifierInterface::calcSelfError()
{
    int numErr  = 0;
    float x,y;
    for(int ii =0; ii<pData.rows; ii++) {
        x   = pData.at<float>(ii,0);
        y   = pData.at<float>(ii,1);
        if(classify(x, y) != lData.at<int>(ii)) {
            numErr++;
        }
    }
    if(pData.rows>0) {
        return (float)numErr/(float)pData.rows;
    } else {
        return -1.f;
    }
}
