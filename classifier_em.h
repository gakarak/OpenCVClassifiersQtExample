#ifndef CLASSIFIER_EM_H
#define CLASSIFIER_EM_H

#include "classifierinterface.h"
#include "classifier_rf.h"
#include <opencv2/legacy/legacy.hpp>

// EM::COV.MATRIX TYPE
static const int em_covmatrixtype_idx[]  = {
    cv::EM::COV_MAT_DIAGONAL,
    cv::EM::COV_MAT_SPHERICAL,
    cv::EM::COV_MAT_GENERIC
};
static const char* em_covmatrixtype_name[] = {
    "EM::COV_MAT_DIAGONAL",
    "EM::COV_MAT_SPHERICAL",
    "EM::COV_MAT_GENERIC"
};
static const int em_covmatrixtype_num  = sizeof(em_covmatrixtype_idx)/sizeof(em_covmatrixtype_idx[0]);

/////////////////////////////////////////////////
class ClassifierEM : public ClassifierInterface
{
public:
    ClassifierEM();
    void    trainData(const std::vector<cv::Point>& data, const std::vector<int>& labels);
    int     classify(int x, int y);
    QString toQString() const;
    //
    void    setParameters(
            int numClusters, int covMatrixType,
            int termCrit_Type, int termCrit_MaxNumIterations, float termCrit_Accuracy);
private:
    cv::ExpectationMaximization  cls;
    int     par_NumClusters;
    int     par_CovMatrixType;
    //
    int     par_TermCrit_Type;
    int     par_TermCrit_MaxNumIterations;
    float   par_TermCrit_Accuracy;
};

#endif // CLASSIFIER_EM_H
