#include "drawlabel.h"

#include <QMouseEvent>
#include <QPaintEvent>
#include <QWheelEvent>

#include <QDebug>
#include <QPainter>
#include <QSet>

DrawLabel::DrawLabel(QWidget *parent) :
    QLabel(parent)
{
    mw  = NULL;
}

void DrawLabel::mousePressEvent(QMouseEvent *ev)
{
    if(mw==NULL) {
        return;
    }
    if(mw->isDrawMode()) {
        switch (ev->button()) {
        case Qt::LeftButton:
            qDebug() << "Qt::LeftButton";
            listPointsOCV.push_back(cv::Point(ev->x(), ev->y()));
            listLabelsOCV.push_back(mw->currentClassIdx);
            update();
            break;
        case Qt::RightButton:
            qDebug() << "Qt::RightButton";
            tryToRemovePoints(ev->x(), ev->y());
            update();
            break;
        case Qt::MiddleButton:
            qDebug() << "Qt::MiddleButton";
            appendClusterOnPosition(ev->x(), ev->y());
            update();
            break;
        default:
            break;
        }
    }
}

void DrawLabel::wheelEvent(QWheelEvent *ev)
{
    if(ev->delta()>0) {
        qDebug() << "+";
        mw->goToPrevClassIdx();
    } else {
        qDebug() << "-";
        mw->goToNextClassIdx();
    }
}

void DrawLabel::paintEvent(QPaintEvent *ev)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    if(mw!=NULL) {
        if(!mw->isDrawMode()) {

        }
    }
    drawPoints(p);
}

void DrawLabel::resizeEvent(QResizeEvent *ev)
{
//    std::vector<cv::Point>  tmpListPointsOCV;
    //    std::vector<int>        tmpListLabelsOCV;
}

void DrawLabel::setMainWindowPtr(MainWindow *mw)
{
    this->mw  = mw;
}

void DrawLabel::setPixmap(const QPixmap &pxm)
{
    this->pxm   = pxm;
}

void DrawLabel::clearPoints()
{
    listPointsOCV.clear();
    listLabelsOCV.clear();
    listExtPoints.clear();
    pxm.fill(Qt::white);
    update();
}

void DrawLabel::drawPoints(QPainter& pnt)
{
    if(mw!=NULL) {
        if(mw->isDrawMode() || (pxm.width()<1) ) {
            pnt.fillRect(rect(), QBrush(Qt::white));
        } else {
            pnt.drawPixmap(0,0, pxm);
        }
        QPen    pen(Qt::black);
        QBrush  brush(Qt::SolidPattern);
        pnt.setPen(pen);
        for(uint ii=0; ii<listPointsOCV.size(); ii++) {
            const cv::Point& p  = listPointsOCV.at(ii);
            const int cIdx      = listLabelsOCV.at(ii);
            brush.setColor(mw->listClassColors.at(cIdx));
            pnt.setBrush(brush);
            pnt.drawEllipse(QRect(p.x-DEF_POINT_RADIUS, p.y-DEF_POINT_RADIUS, 2*DEF_POINT_RADIUS, 2*DEF_POINT_RADIUS));
        }
        // draw ext-points:
        int extRadMax = DEF_POINT_RADIUS-2;
        int extRadMin = DEF_POINT_RADIUS-4;
        for(uint ii=0; ii<listExtPoints.size(); ii++) {
            const cv::Point& p  = listExtPoints.at(ii);
            brush.setColor(Qt::white);
            pnt.setBrush(brush);
            pnt.drawEllipse(QRect(p.x-extRadMax, p.y-extRadMax, 2*extRadMax, 2*extRadMax));
            brush.setColor(Qt::black);
            pnt.setBrush(brush);
            pnt.drawEllipse(QRect(p.x-extRadMin, p.y-extRadMin, 2*extRadMin, 2*extRadMin));
        }
    }
}

void DrawLabel::tryToRemovePoints(int x, int y)
{
    std::vector<cv::Point>  newListPoints;
    std::vector<int>        newListLabels;
    double r2   = DEF_POINT_RADIUS_RM*DEF_POINT_RADIUS_RM;
    for(int ii=0; ii<listPointsOCV.size(); ii++) {
        const cv::Point& p  = listPointsOCV.at(ii);
        double dx   = (double)(x-p.x);
        double dy   = (double)(y-p.y);
        if(r2<(dx*dx+dy*dy)) {
            newListPoints.push_back(listPointsOCV.at(ii));
            newListLabels.push_back(listLabelsOCV.at(ii));
        }
    }
    listPointsOCV   = newListPoints;
    listLabelsOCV   = newListLabels;
}

void DrawLabel::appendClusterOnPosition(int x, int y)
{
    int numCluster  = mw->getClusterSize();
    int sizCluster  = mw->getClusterRadius();
    int cIdx        = mw->currentClassIdx;
    for(int ii=0; ii<numCluster; ii++) {
        int px  = x + cv::theRNG().gaussian(sizCluster);
        int py  = y + cv::theRNG().gaussian(sizCluster);
        listPointsOCV.push_back(cv::Point(px, py));
        listLabelsOCV.push_back(cIdx);
    }
}

void DrawLabel::refreshValidClasses()
{
    if(mw!=NULL) {
        int numOfClass  = mw->getNumberOfClasses();
        std::vector<cv::Point>  newListPoints;
        std::vector<int>        newListLabels;
        for(int ii=0; ii<listPointsOCV.size(); ii++) {
            const cv::Point& pnt    = listPointsOCV.at(ii);
            int cIdx                = listLabelsOCV.at(ii);
            if(cIdx<numOfClass) {
                newListPoints.push_back(pnt);
                newListLabels.push_back(cIdx);
            }
        }
        listPointsOCV.clear();
        listLabelsOCV.clear();
        listPointsOCV   = newListPoints;
        listLabelsOCV   = newListLabels;
    }
}
