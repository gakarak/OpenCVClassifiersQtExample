#ifndef CLASSIFIER_RF_H
#define CLASSIFIER_RF_H

#include "classifierinterface.h"

// RandomForest::TYPE (RandomizedTrees)
static const int rf_termcrit_idx[]  = {
    CV_TERMCRIT_ITER,
    CV_TERMCRIT_EPS,
    CV_TERMCRIT_ITER|CV_TERMCRIT_EPS
};
static const char* rf_termcrit_name[] = {
    "TERMCRIT_ITER",
    "TERMCRIT_EPS",
    "ITER | EPS"
};
static const int rf_termcrit_num  = sizeof(rf_termcrit_idx)/sizeof(rf_termcrit_idx[0]);

/////////////////////////////////////////////////
class ClassifierRF : public ClassifierInterface
{
public:
    ClassifierRF();
    //
    void    trainData(const std::vector<cv::Point>& data, const std::vector<int>& labels);
    int     classify(int x, int y);
    QString toQString() const;
    //
    void    setParameters(
            int maxDepth, int minSampleCount, float regressionAccuracy,
            bool useSurrogates, int maxCategories, bool calcVarImportance,
            int nactiveVars, int maxNumTreesInForest, float forestAccuracy,
            int termCritType);
private:
    cv::RandomTrees cls;
    int         par_MaxDepth;
    int         par_MinSampleCount;
    float       par_RegressionAccuracy;
    bool        par_UseSurrogates;
    int         par_MaxCategories;
    bool        par_CalcVarImportance;
    int         par_NactiveVars;
    int         par_MaxNumTreesInForest;
    float       par_ForestAccuracy;
    int         par_TermCritType;
};

#endif // CLASSIFIER_RF_H
