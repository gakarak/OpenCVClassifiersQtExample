#include "classifier_em.h"

ClassifierEM::ClassifierEM()
{
    par_NumClusters     = cv::EM::DEFAULT_NCLUSTERS;
    par_CovMatrixType   = cv::EM::COV_MAT_DEFAULT;
    //
    par_TermCrit_Type               = rf_termcrit_idx[0];
    par_TermCrit_MaxNumIterations   = 100;
    par_TermCrit_Accuracy           = 0.01f;
}

void ClassifierEM::trainData(const std::vector<cv::Point> &data, const std::vector<int> &labels)
{
    cls.clear();
    loadData(data,labels);
    cv::EMParams params;
    params.covs      = NULL;
    params.means     = NULL;
    params.weights   = NULL;
    params.probs     = NULL;
    params.nclusters        = par_NumClusters;
    params.cov_mat_type     = par_CovMatrixType;
    params.start_step       = cv::EM::START_AUTO_STEP;
    params.term_crit        = cv::TermCriteria(par_TermCrit_Type, par_TermCrit_MaxNumIterations, par_TermCrit_Accuracy);
    cls.train(pData, cv::Mat(), params, &lData);
    isTrainedFlag   = true;
}

int ClassifierEM::classify(int x, int y)
{
    testSample.at<float>(0) = (float)x;
    testSample.at<float>(1) = (float)y;
    return (int)(cls.predict(testSample));
}

QString ClassifierEM::toQString() const
{
    return QString("EM{#cluseters=%1, covMatrixType=%2, Termcrit(type=%3, MaxNumIter=%4, Accuracy=%5)}")
            .arg(par_NumClusters)
            .arg(par_CovMatrixType)
            .arg(par_TermCrit_Type)
            .arg(par_TermCrit_MaxNumIterations)
            .arg(par_TermCrit_Accuracy);
}

void ClassifierEM::setParameters(int numClusters, int covMatrixType, int termCrit_Type, int termCrit_MaxNumIterations, float termCrit_Accuracy)
{
    par_NumClusters     = numClusters;
    par_CovMatrixType   = covMatrixType;
    //
    par_TermCrit_Type               = termCrit_Type;
    par_TermCrit_MaxNumIterations   = termCrit_MaxNumIterations;
    par_TermCrit_Accuracy           = termCrit_Accuracy;
}
