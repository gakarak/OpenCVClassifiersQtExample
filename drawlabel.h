#ifndef DRAWLABEL_H
#define DRAWLABEL_H

#include <QLabel>
#include <QPoint>
#include <QVector>

#include "mainwindow.h"

#include <iostream>
#include <vector>
#include <cv.h>

#define DEF_POINT_RADIUS        5
#define DEF_POINT_RADIUS_RM     10

class DrawLabel : public QLabel
{
    Q_OBJECT
public:
    explicit DrawLabel(QWidget *parent = 0);
    
    void mousePressEvent(QMouseEvent *ev);
    void wheelEvent(QWheelEvent *ev);
    void paintEvent(QPaintEvent *ev);
    void resizeEvent(QResizeEvent *ev);

    std::vector<cv::Point>  listPointsOCV;
    std::vector<int>        listLabelsOCV;
    std::vector<cv::Point>  listExtPoints;
    void setMainWindowPtr(MainWindow *mw);
    void setPixmap(const QPixmap &pxm);
    void refreshValidClasses();

signals:
    
public slots:
    void clearPoints();

private:
    void drawPoints(QPainter &p);
    void tryToRemovePoints(int x, int y);
    void appendClusterOnPosition(int x, int y);
    MainWindow*     mw;
    QPixmap         pxm;
    
};

#endif // DRAWLABEL_H
