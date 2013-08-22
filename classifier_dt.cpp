#include "classifier_dt.h"

ClassifierDT::ClassifierDT()
{
    par_MaxDepth            = 16;
    par_MinSampleCount       = 2;
    par_IsUseSurrogates     = true;
    par_IsUse1seRule        = true;
    par_IsTrucatePrunedTree = true;
    par_NumValidationFolds  = 0;
    par_RegressionAccuracy  = 0.01f;
}

void ClassifierDT::trainData(const std::vector<cv::Point> &data, const std::vector<int> &labels)
{
    cls.clear();
    loadData(data, labels);
    cv::Mat var_types( 1, pData.cols + 1, CV_8UC1, cv::Scalar(CV_VAR_ORDERED) );
    var_types.at<uchar>( pData.cols ) = CV_VAR_CATEGORICAL;
    cv::DTreeParams params;
    params.max_depth            = par_MaxDepth;
    params.min_sample_count     = par_MinSampleCount;
    params.use_surrogates       = par_IsUseSurrogates;
    params.use_1se_rule         = par_IsUse1seRule;
    params.truncate_pruned_tree = par_IsTrucatePrunedTree;
    params.cv_folds             = par_NumValidationFolds;
    params.regression_accuracy  = par_RegressionAccuracy;
    cls.train(pData, CV_ROW_SAMPLE, lData, cv::Mat(), cv::Mat(), var_types, cv::Mat(), params);
    isTrainedFlag   = true;
}

int ClassifierDT::classify(int x, int y)
{
    testSample.at<float>(0) = (float)x;
    testSample.at<float>(1) = (float)y;
    return cvRound(cls.predict(testSample)->value);
}

QString ClassifierDT::toQString() const
{
    return QString("DT{max_depth=%1, min_sample_count=%2, use_surrogates=%3, use_1se_rule=%4, truncate_pruned_tree=%5, cv_folds=%6, regression_accuracy=%7}")
            .arg(par_MaxDepth)
            .arg(par_MinSampleCount)
            .arg(par_IsUseSurrogates)
            .arg(par_IsUse1seRule)
            .arg(par_IsTrucatePrunedTree)
            .arg(par_NumValidationFolds)
            .arg(par_RegressionAccuracy);
}

void ClassifierDT::setParameters(int maxDepth, int minSampleCount, bool isUseSurrogates, bool isUse1seRule, bool isTrucatePrunedTree, int numValidationFolds, float regressionAccuracy)
{
    par_MaxDepth            = maxDepth;
    par_MinSampleCount      = minSampleCount;
    par_IsUseSurrogates     = isUseSurrogates;
    par_IsUse1seRule        = isUse1seRule;
    par_IsTrucatePrunedTree = isTrucatePrunedTree;
    par_NumValidationFolds  = numValidationFolds;
    par_RegressionAccuracy  = regressionAccuracy;
}
