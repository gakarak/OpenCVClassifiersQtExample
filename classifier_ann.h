#ifndef CLASSIFIER_ANN_H
#define CLASSIFIER_ANN_H

#include "classifierinterface.h"
#include "classifier_rf.h"

// ANN::METHOD
static const int ann_method_idx[]  = {
    cv::ANN_MLP_TrainParams::BACKPROP,
    cv::ANN_MLP_TrainParams::RPROP
};
static const char* ann_method_name[] = {
    "ANN::BACKPROP",
    "ANN::RPROP"
};
static const int ann_method_num  = sizeof(ann_method_idx)/sizeof(ann_method_idx[0]);

// ANN::F(X)
static const int ann_fx_idx[]  = {
    cv::NeuralNet_MLP::SIGMOID_SYM,
    cv::NeuralNet_MLP::GAUSSIAN
};
static const char* ann_fx_name[] = {
    "ANN::SIGMOID_SYM",
    "ANN::GAUSSIAN"
};
static const int ann_fx_num  = sizeof(ann_fx_idx)/sizeof(ann_fx_idx[0]);

/////////////////////////////////////////////////
class ClassifierANN : public ClassifierInterface
{
public:
    ClassifierANN();
    //
    void    trainData(const std::vector<cv::Point>& data, const std::vector<int>& labels);
    int     classify(int x, int y);
    QString toQString() const;
    //
    void    setParameters(
            int numOfClasses,
            int typeMethod, int typeFx,
            double  param1, double param2,
            int numNeurons_Layer1, int numNeurons_Layer2, int numNeurons_Layer3,
            int termCrit_Type, int termCrit_MaxNumIterations, float termCrit_Accuracy);
private:
    cv::NeuralNet_MLP cls;
    cv::Mat weights;
    cv::Mat layerSizes;
    cv::Mat getLayerSizes();
    cv::Mat getOutputClasses();
    //
    int     par_NumOfClasses;
    int     par_TypeMethod;
    int     par_TypeFx;
    double  par_Param1;
    double  par_Param2;
    //
    int     par_NumNeurons_Layer1;
    int     par_NumNeurons_Layer2;
    int     par_NumNeurons_Layer3;
    //
    int     par_TermCrit_Type;
    int     par_TermCrit_MaxNumIterations;
    float   par_TermCrit_Accuracy;
};

#endif // CLASSIFIER_ANN_H
