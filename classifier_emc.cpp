#include "classifier_emc.h"

ClassifierEMC::ClassifierEMC()
{
    par_NumOfClasses    = -1;
    par_NumClusters     = cv::EM::DEFAULT_NCLUSTERS;
    par_CovMatrixType   = cv::EM::COV_MAT_DEFAULT;
    //
    par_TermCrit_Type               = rf_termcrit_idx[0];
    par_TermCrit_MaxNumIterations   = 100;
    par_TermCrit_Accuracy           = 0.01f;
}

void ClassifierEMC::trainData(const std::vector<cv::Point> &data, const std::vector<int> &labels)
{
    cls.clear();
    loadData(data,labels);
    //
    if(par_NumOfClasses<1) {
        std::cerr << "Bad initialization of #classes" << std::endl;
        return;
    }
    cls.resize(par_NumOfClasses);
    for(uint idx=0; idx<cls.size(); idx++) {
        cls[idx]    = cv::EM(par_NumClusters, par_CovMatrixType, cv::TermCriteria(par_TermCrit_Type, par_TermCrit_MaxNumIterations, par_TermCrit_Accuracy));
        cv::Mat modelSamples;
        for(int ii=0; ii<lData.rows; ii++) {
            if((uint)lData.at<int>(ii)==idx) {
                modelSamples.push_back(pData.row(ii));
            }
        }
        if(!modelSamples.empty()) {
            cls[idx].train(modelSamples);
        }
    }
    isTrainedFlag   = true;
}

int ClassifierEMC::classify(int x, int y)
{
    testSample.at<float>(0) = (float)x;
    testSample.at<float>(1) = (float)y;
    cv::Mat logLikelihoods(1, cls.size(), CV_64FC1, cv::Scalar(-DBL_MAX));
    for(uint ii=0; ii<cls.size(); ii++) {
        if(cls[ii].isTrained()) {
            logLikelihoods.at<double>(ii) = cls[ii].predict(testSample)[0];
        }
    }
    cv::Point maxLoc;
    minMaxLoc(logLikelihoods, 0, 0, 0, &maxLoc);
    int ret = maxLoc.x;
    if((ret<0)||(ret>=par_NumOfClasses)) {
        ret=0;
    }
    return ret;
}

QString ClassifierEMC::toQString() const
{
    return QString("EM{#cluseters=%1, covMatrixType=%2, Termcrit(type=%3, MaxNumIter=%4, Accuracy=%5), #cls=%6}")
            .arg(par_NumClusters)
            .arg(par_CovMatrixType)
            .arg(par_TermCrit_Type)
            .arg(par_TermCrit_MaxNumIterations)
            .arg(par_TermCrit_Accuracy)
            .arg(par_NumOfClasses);
}

void ClassifierEMC::setParameters(int numOfClasses, int numClusters, int covMatrixType, int termCrit_Type, int termCrit_MaxNumIterations, float termCrit_Accuracy)
{
    par_NumOfClasses    = numOfClasses;
    par_NumClusters     = numClusters;
    par_CovMatrixType   = covMatrixType;
    //
    par_TermCrit_Type               = termCrit_Type;
    par_TermCrit_MaxNumIterations   = termCrit_MaxNumIterations;
    par_TermCrit_Accuracy           = termCrit_Accuracy;
}
