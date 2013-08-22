#ifndef CLASSIFIER_BT_H
#define CLASSIFIER_BT_H

#include "classifierinterface.h"

// BOOST::TYPE
static const int bt_type_idx[]  = {
    cv::Boost::DISCRETE,
    cv::Boost::REAL,
    cv::Boost::LOGIT,
    cv::Boost::GENTLE,

};
static const char* bt_type_name[] = {
    "Boost::DISCRETE",
    "Boost::REAL",
    "Boost::LOGIT",
    "Boost::GENTLE",
};
static const int bt_type_num  = sizeof(bt_type_idx)/sizeof(bt_type_idx[0]);

/////////////////////////////////////////////////
class ClassifierBT : public ClassifierInterface
{
public:
    ClassifierBT();
    //
    void    trainData(const std::vector<cv::Point>& data, const std::vector<int>& labels);
    int     classify(int x, int y);
    QString toQString() const;
    int     getMaxClassNum() {
        return 2;
    }
    //
    void    setParameters(int type, int weakCount, double weightTrimRate, int maxDepth, bool useSurrogates);
private:
    cv::Boost   cls;
    int         par_Type;
    int         par_WeakCount;
    double      par_WeightTrimRate;
    int         par_MaxDepth;
    bool        par_IsUseSurrogates;

};

#endif // CLASSIFIER_BT_H
