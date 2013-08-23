#-------------------------------------------------
#
# Project created by QtCreator 2013-08-19T16:23:29
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = OpenCVClassifiersQtExample

TEMPLATE = app

unix {
    INCLUDEPATH += /home/ar/dev/opencv-2.4.6.1-cuda-release/include/opencv /home/ar/dev/opencv-2.4.6.1-cuda-release/include
    LIBS        += -L/home/ar/dev/opencv-2.4.6.1-cuda-release/lib  -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab
}


SOURCES += main.cpp \
    mainwindow.cpp \
    drawlabel.cpp \
    classifier_nbc.cpp \
    classifierinterface.cpp \
    classifier_knn.cpp \
    classifier_svm.cpp \
    classifier_dt.cpp \
    classifier_bt.cpp \
    classifier_gbt.cpp \
    classifier_rf.cpp \
    classifier_ert.cpp \
    classifier_ann.cpp \
    classifier_em.cpp \
    classifier_emc.cpp

HEADERS += \
    mainwindow.h \
    drawlabel.h \
    classifierinterface.h \
    classifier_nbc.h \
    classifier_knn.h \
    classifier_svm.h \
    classifier_dt.h \
    classifier_bt.h \
    classifier_gbt.h \
    classifier_rf.h \
    classifier_ert.h \
    classifier_ann.h \
    classifier_em.h \
    classifier_emc.h

FORMS += \
    mainwindow.ui

RESOURCES += \
    imgres.qrc
