#ifndef CLASSIFIER_ERT_H
#define CLASSIFIER_ERT_H

#include "classifierinterface.h"
#include "classifier_rf.h"

/////////////////////////////////////////////////
class ClassifierERT : public ClassifierInterface
{
public:
    ClassifierERT();
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
    cv::ERTrees cls;
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

#endif // CLASSIFIER_ERT_H
