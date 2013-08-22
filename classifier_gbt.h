#ifndef CLASSIFIER_GBT_H
#define CLASSIFIER_GBT_H

#include "classifierinterface.h"

// GradientBoostedTrees::TYPE
static const int gbt_type_idx[]  = {
    cv::GradientBoostingTrees::DEVIANCE_LOSS,
    cv::GradientBoostingTrees::SQUARED_LOSS,
    cv::GradientBoostingTrees::ABSOLUTE_LOSS,
    cv::GradientBoostingTrees::HUBER_LOSS
};
static const char* gbt_type_name[] = {
    "GBT::DEVIANCE_LOSS",
    "GBT::SQUARED_LOSS",
    "GBT::ABSOLUTE_LOSS",
    "GBT::HUBER_LOSS",
};
static const int gbt_type_num  = sizeof(gbt_type_idx)/sizeof(gbt_type_idx[0]);

/////////////////////////////////////////////////
class ClassifierGBT : public ClassifierInterface
{
public:
    ClassifierGBT();
    void    trainData(const std::vector<cv::Point>& data, const std::vector<int>& labels);
    int     classify(int x, int y);
    QString toQString() const;
    //
    void    setParameters(int type, int weakCount, float shrinkage, float subsamplePortion, int maxDepth, bool useSurrogates);
private:
    cv::GradientBoostingTrees   cls;
    int         par_Type;
    int         par_WeakCount;
    float       par_Shrinkage;
    float       par_SubsamplePortion;
    int         par_MaxDepth;
    bool        par_UseSurrogates;
};

#endif // CLASSIFIER_GBT_H
