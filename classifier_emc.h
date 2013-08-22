#ifndef CLASSIFIER_EMC_H
#define CLASSIFIER_EMC_H

#include "classifierinterface.h"
#include "classifier_em.h"

class ClassifierEMC : public ClassifierInterface
{
public:
    ClassifierEMC();
    void    trainData(const std::vector<cv::Point>& data, const std::vector<int>& labels);
    int     classify(int x, int y);
    QString toQString() const;
    //
    void    setParameters(
            int numOfClasses,
            int numClusters, int covMatrixType,
            int termCrit_Type, int termCrit_MaxNumIterations, float termCrit_Accuracy);
private:
    std::vector<cv::EM> cls;
    int     par_NumOfClasses;
    int     par_NumClusters;
    int     par_CovMatrixType;
    //
    int     par_TermCrit_Type;
    int     par_TermCrit_MaxNumIterations;
    float   par_TermCrit_Accuracy;
};

#endif // CLASSIFIER_EM_H
