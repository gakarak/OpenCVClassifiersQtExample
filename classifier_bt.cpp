#include "classifier_bt.h"

ClassifierBT::ClassifierBT()
{
    par_Type            = bt_type_idx[0];
    par_WeakCount       = 100;
    par_WeightTrimRate  = 0.95;
    par_MaxDepth        = 2;
    par_IsUseSurrogates = false;
}

void ClassifierBT::trainData(const std::vector<cv::Point> &data, const std::vector<int> &labels)
{
    cls.clear();
    loadData(data, labels);
    cv::Mat var_types( 1, pData.cols + 1, CV_8UC1, cv::Scalar(CV_VAR_ORDERED) );
    var_types.at<uchar>( pData.cols ) = CV_VAR_CATEGORICAL;
    cv::BoostParams params;
    params.boost_type       = par_Type;
    params.weak_count       = par_WeakCount;
    params.weight_trim_rate = par_WeightTrimRate;
    params.max_depth        = par_MaxDepth;
    params.use_surrogates   = par_IsUseSurrogates;
    cls.train(pData, CV_ROW_SAMPLE, lData, cv::Mat(), cv::Mat(), var_types, cv::Mat(), params);
    isTrainedFlag   = true;
}

int ClassifierBT::classify(int x, int y)
{
    testSample.at<float>(0) = (float)x;
    testSample.at<float>(1) = (float)y;
    return cvRound(cls.predict(testSample));
}

QString ClassifierBT::toQString() const
{
    return QString("BT(boost_type=%1, weak_count=%2, weight_trim_rate=%3, max_depth=%4, use_surrogates=%5)")
            .arg(bt_type_name[par_Type])
            .arg(par_WeakCount)
            .arg(par_WeightTrimRate)
            .arg(par_MaxDepth)
            .arg(par_IsUseSurrogates);
}

void ClassifierBT::setParameters(int type, int weakCount, double weightTrimRate, int maxDepth, bool useSurrogates)
{
    par_Type            = type;
    par_WeakCount       = weakCount;
    par_WeightTrimRate  = weightTrimRate;
    par_MaxDepth        = maxDepth;
    par_IsUseSurrogates = useSurrogates;
}
