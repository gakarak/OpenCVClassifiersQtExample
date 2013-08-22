#include "classifier_svm.h"

ClassifierSVM::ClassifierSVM()
{
    par_Type    = svm_type_idx[0];
    par_Kernel  = svm_kernel_idx[0];
    par_Degree  = 0.5;
    par_Gamma   = 1;
    par_Coef0   = 1;
    par_C       = 1;
    par_Nu      = 0.5;
}

void ClassifierSVM::setParameters(int type, int kernel, double degree, double gamma, double coef0, double c, double nu)
{
    par_Type    = type;
    par_Kernel  = kernel;
    par_Degree  = degree;
    par_Gamma   = gamma;
    par_Coef0   = coef0;
    par_C       = c;
    par_Nu      = nu;
}

QString ClassifierSVM::toQString() const
{
    return QString("SVM{type=%1, kernel=%2, Degree=%3, Gamma=%4, Coef0=%5, C=%6, Nu=%7}")
            .arg(svm_type_name[0])
            .arg(svm_kernel_name[0])
            .arg(par_Degree)
            .arg(par_Gamma)
            .arg(par_Coef0)
            .arg(par_C)
            .arg(par_Nu);
}

void ClassifierSVM::loadAdditionalPoints(std::vector<cv::Point> &extPoints)
{
    extPoints.clear();
    if(isTrainedFlag) {
        for(int ii=0; ii<cls.get_support_vector_count(); ii++) {
            const float* sv = cls.get_support_vector(ii);
            extPoints.push_back(cv::Point(sv[0], sv[1]));
        }
    }
}

int ClassifierSVM::classify(int x, int y)
{
    testSample.at<float>(0) = (float)x;
    testSample.at<float>(1) = (float)y;
    return cvRound(cls.predict(testSample));
}

void ClassifierSVM::trainData(const std::vector<cv::Point> &data, const std::vector<int> &labels)
{
    cls.clear();
    loadData(data, labels);
    cv::SVMParams params;
    params.svm_type     = par_Type;
    params.kernel_type  = par_Kernel;
    params.degree       = par_Degree;
    params.gamma        = par_Gamma;
    params.coef0        = par_Coef0;
    params.C            = par_C;
    params.nu           = par_Nu;
    params.p            = 0;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
    //
    cls.train(pData, lData, cv::Mat(), cv::Mat(), params);
    isTrainedFlag   = true;
}


