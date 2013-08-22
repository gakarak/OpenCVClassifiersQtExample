#ifndef CLASSIFIER_SVM_H
#define CLASSIFIER_SVM_H

#include "classifierinterface.h"

////////////////////////////////////////////////////////
// SVM::TYPE
static const int svm_type_idx[]  = {
    cv::SVM::C_SVC,
    cv::SVM::NU_SVC,
    cv::SVM::ONE_CLASS,
    cv::SVM::EPS_SVR,
    cv::SVM::NU_SVR

};
static const char* svm_type_name[] = {
    "SVM::C_SVC",
    "SVM::NU_SVC",
    "SVM::ONE_CLASS",
    "SVM::EPS_SVR",
    "SVM::NU_SVR"
};
static const int svm_type_num  = sizeof(svm_type_idx)/sizeof(svm_type_idx[0]);

// SVM::KERNEL
static const int svm_kernel_idx[]  = {
    cv::SVM::LINEAR,
    cv::SVM::POLY,
    cv::SVM::RBF,
    cv::SVM::SIGMOID
};
static const char* svm_kernel_name[] = {
    "SVM::LINEAR",
    "SVM::POLY",
    "SVM::RBF",
    "SVM::SIGMOID"
};
static const int svm_kernel_num  = sizeof(svm_kernel_idx)/sizeof(svm_kernel_idx[0]);

////////////////////////////////////////////////////////
class ClassifierSVM : public ClassifierInterface
{
public:
    ClassifierSVM();
    //
    virtual void    trainData(const std::vector<cv::Point>& data, const std::vector<int>& labels);
    virtual int     classify(int x, int y);
    virtual QString toQString() const;
    virtual void    loadAdditionalPoints(std::vector<cv::Point>& extPoints);
    //
    void setParameters(int type, int kernel, double degree, double gamma, double coef0, double c, double nu);
private:
    int     par_Type;
    int     par_Kernel;
    double  par_Degree;
    double  par_Gamma;
    double  par_Coef0;
    double  par_C;
    double  par_Nu;
    //
    cv::SVM cls;
};

#endif // CLASSIFIER_SVM_H
