#ifndef CLASSIFIER_DT_H
#define CLASSIFIER_DT_H

#include "classifierinterface.h"

class ClassifierDT : public ClassifierInterface
{
public:
    ClassifierDT();
    //
    void    trainData(const std::vector<cv::Point>& data, const std::vector<int>& labels);
    int     classify(int x, int y);
    QString toQString() const;
    //
    void setParameters(
            int maxDepth, int minSampleCount,
            bool isUseSurrogates, bool isUse1seRule,
            bool isTrucatePrunedTree, int numValidationFolds,
            float regressionAccuracy);
private:
    int     par_MaxDepth;
    int     par_MinSampleCount;
    bool    par_IsUseSurrogates;
    bool    par_IsUse1seRule;
    bool    par_IsTrucatePrunedTree;
    int     par_NumValidationFolds;
    float   par_RegressionAccuracy;
    cv::DecisionTree cls;
};

#endif // CLASSIFIER_DT_H
