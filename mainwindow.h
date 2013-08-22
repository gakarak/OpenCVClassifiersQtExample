#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVector>
#include <QIcon>

#include "classifierinterface.h"

#define DEF_NUMCLASS_MIN    2
#define DEF_NUMCLASS_MAX    10

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    QVector<QIcon>  listClassIcons;
    QVector<QColor> listClassColors;
    int             currentClassIdx;
    int             currentClassNum;

    void    goToNextClassIdx();
    void    goToPrevClassIdx();
    int     getNumberOfClasses() const;
    int     getClusterSize() const;
    int     getClusterRadius() const;
    bool    isDrawMode() const;

    std::vector<ClassifierInterface* > listOfClassifiers;


private slots:
    void on_spinBoxNumberOfClasses_valueChanged(int num);
    void on_pushButtonRefreshClassColors_clicked();

    void on_pushButtonApplyClassifier_clicked();

    void on_comboBoxCurrentClassIdx_currentIndexChanged(int index);

    void on_pushButtonClearPoints_clicked();

    void on_pushButtonTrainClassifier_clicked();

    void on_checkBoxIsDrawMode_clicked();

    void on_action_Exit_triggered();

    void on_actionFullscreen_triggered();

    void on_toolButton_clicked();

    void on_checkBox_ANN_SizeLayer2_clicked();

    void on_checkBox_ANN_SizeLayer3_clicked();

private:
    Ui::MainWindow *ui;

    void setNumberOfClasses(int numberOfClasses);
    void setCurrentClassIdx(int currentClassIdx);
    void buildListClassColors();

};

#endif // MAINWINDOW_H
