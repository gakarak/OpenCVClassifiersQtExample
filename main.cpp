//#include <QCoreApplication>
#include <QApplication>

/*
#include <iostream>
#include <vector>
#include <cv.h>
#include <highgui.h>
*/


using namespace std;

#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();

/*
    vector<cv::Point>   data;
    vector<int>         labels;
    for(int ii=0; ii<8; ii++) {
        data.push_back(cv::Point(ii,ii));
        labels.push_back(ii);
    }
    cv::Mat retData, retLabels;
    cv::Mat(data).copyTo(retData);
    retData = retData.reshape(1, retData.rows);
    retData.convertTo(retData, CV_32FC1);
    cv::Mat(labels).copyTo(retLabels);
    cout << "(" << retData.cols   << "x" << retData.rows << ":" << retData.channels()  << ") = " << retData   << endl;
    cout << "(" << retLabels.cols << "x" << retLabels.rows << ":" << retLabels.channels() << ") = " << retLabels << endl;
*/
}
