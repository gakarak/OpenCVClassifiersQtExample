#include "classifier_rf.h"

ClassifierRF::ClassifierRF()
{
    par_MaxDepth            = 4;
    par_MinSampleCount      = 2;
    par_RegressionAccuracy  = 0.f;
    par_UseSurrogates       = false;
    par_MaxCategories       = 16;
    par_CalcVarImportance   = false;
    par_NactiveVars         = 1;
    par_MaxNumTreesInForest = 5;
    par_ForestAccuracy      = 0.f;
    par_TermCritType        = rf_termcrit_idx[0];
}

void ClassifierRF::trainData(const std::vector<cv::Point> &data, const std::vector<int> &labels)
{
    cls.clear();
    loadData(data, labels);
    cv::RandomTreeParams params;
    params.max_depth            = par_MaxDepth;
    params.min_sample_count     = par_MinSampleCount;
    params.regression_accuracy  = par_RegressionAccuracy;
    params.use_surrogates       = par_UseSurrogates;
    params.max_categories       = par_MaxCategories;
    params.calc_var_importance  = par_CalcVarImportance;
    params.nactive_vars         = par_NactiveVars;
    params.term_crit            = cv::TermCriteria(par_TermCritType, par_MaxNumTreesInForest, par_ForestAccuracy);
    cls.train(pData, CV_ROW_SAMPLE, lData, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), params);
    isTrainedFlag   = true;
}

int ClassifierRF::classify(int x, int y)
{
    testSample.at<float>(0) = (float)x;
    testSample.at<float>(1) = (float)y;
    return cvRound(cls.predict(testSample));
}

QString ClassifierRF::toQString() const
{
    return QString("RF{max_depth=%1, min_sample_count=%2, regression_accuracy=%3, use_surrogates=%4, max_categories=%5, calc_var_importance=%6, nactive_vars==%7, termCrit=(%8,%9,%10)}")
            .arg(par_MaxDepth)
            .arg(par_MinSampleCount)
            .arg(par_RegressionAccuracy)
            .arg(par_UseSurrogates)
            .arg(par_MaxCategories)
            .arg(par_CalcVarImportance)
            .arg(par_NactiveVars)
            .arg(par_TermCritType)
            .arg(par_MaxNumTreesInForest)
            .arg(par_ForestAccuracy);
}

void ClassifierRF::setParameters(int maxDepth, int minSampleCount, float regressionAccuracy, bool useSurrogates, int maxCategories, bool calcVarImportance, int nactiveVars, int maxNumTreesInForest, float forestAccuracy, int termCritType)
{
    par_MaxDepth            = maxDepth;
    par_MinSampleCount      = minSampleCount;
    par_RegressionAccuracy  = regressionAccuracy;
    par_UseSurrogates       = useSurrogates;
    par_MaxCategories       = maxCategories;
    par_CalcVarImportance   = calcVarImportance;
    par_NactiveVars         = nactiveVars;
    par_MaxNumTreesInForest = maxNumTreesInForest;
    par_ForestAccuracy      = forestAccuracy;
    par_TermCritType        = termCritType;
}


