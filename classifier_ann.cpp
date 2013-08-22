#include "classifier_ann.h"

ClassifierANN::ClassifierANN()
{
    par_NumOfClasses    = -1;
    par_TypeMethod      = ann_method_idx[0];
    par_TypeFx          = ann_fx_idx[0];
    par_Param1          = 0.1;
    par_Param2          = 0.1;
    //
    par_NumNeurons_Layer1   = 8;
    par_NumNeurons_Layer2   = -1;
    par_NumNeurons_Layer3   = -1;
    //
    par_TermCrit_Type               = rf_termcrit_idx[0];
    par_TermCrit_MaxNumIterations   = 100;
    par_TermCrit_Accuracy           = 0.01f;
}

void ClassifierANN::trainData(const std::vector<cv::Point> &data, const std::vector<int> &labels)
{
//    cls.clear();
    //
    if(par_NumOfClasses<0) {
        std::cerr << "Bad initialization of #classes" << std::endl;
        return;
    }
    loadData(data, labels);
    cv::Mat outClasses  = cv::Mat::zeros(data.size(), par_NumOfClasses, CV_32FC1);
    for(int ss=0; ss<outClasses.rows; ss++) {
        int lidx    = labels.at(ss);
        for(int ii=0; ii<outClasses.cols; ii++) {
            if(ii==lidx) {
                outClasses.at<float>(ss,ii) = 1.f;
            } else {
                outClasses.at<float>(ss,ii) = 0.f;
            }
        }
    }
    //
    if(par_NumNeurons_Layer3>0) {
        layerSizes  = cv::Mat(1,5, CV_32SC1);
        layerSizes.at<int>(0) = 2;
        layerSizes.at<int>(1) = par_NumNeurons_Layer1;
        layerSizes.at<int>(2) = par_NumNeurons_Layer2;
        layerSizes.at<int>(3) = par_NumNeurons_Layer3;
        layerSizes.at<int>(4) = par_NumOfClasses;
    } else {
        if(par_NumNeurons_Layer2>0) {
            layerSizes  = cv::Mat(1,4, CV_32SC1);
            layerSizes.at<int>(0) = 2;
            layerSizes.at<int>(1) = par_NumNeurons_Layer1;
            layerSizes.at<int>(2) = par_NumNeurons_Layer2;
            layerSizes.at<int>(3) = par_NumOfClasses;
        } else {
            layerSizes  = cv::Mat(1,3, CV_32SC1);
            layerSizes.at<int>(0) = 2;
            layerSizes.at<int>(1) = par_NumNeurons_Layer1;
            layerSizes.at<int>(2) = par_NumOfClasses;
        }
    }
    weights = cv::Mat( 1, data.size(), CV_32FC1, cv::Scalar::all(1) );
//    std::cout << "pData      :" << pData      << std::endl;
//    std::cout << "lData      :" << lData      << std::endl;
//    std::cout << "outClasses :" << outClasses << std::endl;
//    std::cout << "layers     :" << layerSizes << std::endl;
//    std::cout << "weights    :" << weights    << std::endl;
    cv::ANN_MLP_TrainParams params  = cv::ANN_MLP_TrainParams(
                cv::TermCriteria(par_TermCrit_Type, par_TermCrit_MaxNumIterations, par_TermCrit_Accuracy),
                par_TypeMethod,
                par_Param1,
                par_Param2);
    cls.create(layerSizes, par_TypeFx, par_Param1, par_Param2);
    cls.train(pData, outClasses, weights, cv::Mat(), params);
//    cls.create(layerSizes, cv::NeuralNet_MLP::SIGMOID_SYM, 1, 1);
//    cls.train(pData,outClasses, weights);
    isTrainedFlag   = true;
}

int ClassifierANN::classify(int x, int y)
{
    testSample.at<float>(0) = (float)x;
    testSample.at<float>(1) = (float)y;
    cv::Mat out = cv::Mat::zeros(1, par_NumOfClasses, CV_32FC1);
    cls.predict(testSample, out);
//    std::cout << out << std::endl;
    cv::Point pMax;
    cv::minMaxLoc(out, 0, 0, 0, &pMax);
    int ret = pMax.x;
    if((ret<0)||(ret>=par_NumOfClasses)) {
        ret=0;
    }
    return ret;
}

QString ClassifierANN::toQString() const
{
    return QString("ANN{method=%1, Fx=%2, param1=%3, param2=%4, #neurons={%5,%6,%7}, TermCrit(type=%8, #iter=%9, accuracy=%10)}")
            .arg(par_TypeMethod)
            .arg(par_TypeFx)
            .arg(par_Param1)
            .arg(par_Param2)
            .arg(par_NumNeurons_Layer1).arg(par_NumNeurons_Layer2).arg(par_NumNeurons_Layer3)
            .arg(par_TermCrit_Type)
            .arg(par_TermCrit_MaxNumIterations)
            .arg(par_TermCrit_Accuracy);
}

void ClassifierANN::setParameters(
        int numOfClasses,
        int typeMethod, int typeFx, double param1, double param2,
        int numNeurons_Layer1, int numNeurons_Layer2, int numNeurons_Layer3,
        int termCrit_Type, int termCrit_MaxNumIterations, float termCrit_Accuracy)
{
    par_NumOfClasses    = numOfClasses;
    par_TypeMethod      = typeMethod;
    par_TypeFx          = typeFx;
    par_Param1          = param1;
    par_Param2          = param2;
    //
    par_NumNeurons_Layer1   = numNeurons_Layer1;
    par_NumNeurons_Layer2   = numNeurons_Layer2;
    par_NumNeurons_Layer3   = numNeurons_Layer3;
    //
    par_TermCrit_Type               = termCrit_Type;
    par_TermCrit_MaxNumIterations   = termCrit_MaxNumIterations;
    par_TermCrit_Accuracy           = termCrit_Accuracy;
}

