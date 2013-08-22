#include "classifier_gbt.h"

ClassifierGBT::ClassifierGBT()
{
    par_Type                = gbt_type_idx[0];
    par_WeakCount           = 100;
    par_Shrinkage           = 0.1;
    par_SubsamplePortion    = 0.2;
    par_MaxDepth            = 2;
    par_UseSurrogates       = false;
}

void ClassifierGBT::trainData(const std::vector<cv::Point> &data, const std::vector<int> &labels)
{
    cls.clear();
    loadData(data, labels);
    cv::Mat var_types( 1, pData.cols + 1, CV_8UC1, cv::Scalar(CV_VAR_ORDERED) );
    var_types.at<uchar>( pData.cols ) = CV_VAR_CATEGORICAL;
    cv::GradientBoostingTreeParams params;
    params.loss_function_type   = par_Type;
    params.weak_count           = par_WeakCount;
    params.shrinkage            = par_Shrinkage;
    params.subsample_portion    = par_SubsamplePortion;
    params.max_depth            = par_MaxDepth;
    params.use_surrogates       = par_UseSurrogates;
    cls.train(pData, CV_ROW_SAMPLE, lData, cv::Mat(), cv::Mat(), var_types, cv::Mat(), params);
    isTrainedFlag   = true;
}

int ClassifierGBT::classify(int x, int y)
{
    testSample.at<float>(0) = (float)x;
    testSample.at<float>(1) = (float)y;
    return cvRound(cls.predict(testSample));
}

QString ClassifierGBT::toQString() const
{
    return QString("GBT{loss_fun_type=%1, weak_count=%2, shrinkage=%3, subsample_portion=%4, max_depth=%5, use_surrogates=%6}")
            .arg(gbt_type_name[par_Type])
            .arg(par_WeakCount)
            .arg(par_Shrinkage)
            .arg(par_SubsamplePortion)
            .arg(par_MaxDepth)
            .arg(par_UseSurrogates);
}

void ClassifierGBT::setParameters(int type, int weakCount, float shrinkage, float subsamplePortion, int maxDepth, bool useSurrogates)
{
    par_Type                = type;
    par_WeakCount           = weakCount;
    par_Shrinkage           = shrinkage;
    par_SubsamplePortion    = subsamplePortion;
    par_MaxDepth            = maxDepth;
    par_UseSurrogates       = useSurrogates;
}
