#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>

#include <QImage>
#include <QPixmap>
#include <QPainter>
#include <QUrl>

#include <QMessageBox>
#include <QDesktopServices>

#include "classifier_nbc.h"
#include "classifier_knn.h"
#include "classifier_svm.h"
#include "classifier_dt.h"
#include "classifier_bt.h"
#include "classifier_gbt.h"
#include "classifier_rf.h"
#include "classifier_ert.h"
#include "classifier_ann.h"
#include "classifier_em.h"
#include "classifier_emc.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    buildListClassColors();
    ui->progressBar->setValue(0);
    //
    ui->spinBoxNumberOfClasses->setMinimum(DEF_NUMCLASS_MIN);
    ui->spinBoxNumberOfClasses->setMaximum(DEF_NUMCLASS_MAX);
    ui->spinBoxNumberOfClasses->setValue(DEF_NUMCLASS_MIN);
    //
    setNumberOfClasses(ui->spinBoxNumberOfClasses->value());
    //
    ui->label->setMainWindowPtr(this);
    //
    listOfClassifiers.push_back(new ClassifierNBC());
    listOfClassifiers.push_back(new ClassifierKNN());
    listOfClassifiers.push_back(new ClassifierSVM());
    listOfClassifiers.push_back(new ClassifierDT ());
    listOfClassifiers.push_back(new ClassifierBT ());
    listOfClassifiers.push_back(new ClassifierGBT());
    listOfClassifiers.push_back(new ClassifierRF ());
    listOfClassifiers.push_back(new ClassifierERT());
    listOfClassifiers.push_back(new ClassifierANN());
    listOfClassifiers.push_back(new ClassifierEM ());
    listOfClassifiers.push_back(new ClassifierEMC());
    //
    ui->checkBoxIsDrawMode->setChecked(true);
    on_checkBoxIsDrawMode_clicked();
    // [SVM]
    for(int ii=0; ii<svm_type_num; ii++) {
        ui->comboBox_SVM_Type->addItem(QString("%1").arg(svm_type_name[ii]));
    }
    ui->comboBox_SVM_Type->setCurrentIndex(0);
    for(int ii=0; ii<svm_kernel_num; ii++) {
        ui->comboBox_SVM_Kernel->addItem(QString("%1").arg(svm_kernel_name[ii]));
    }
    ui->comboBox_SVM_Kernel->setCurrentIndex(0);
    // [BT]
    for(int ii=0; ii<bt_type_num; ii++) {
        ui->comboBox_BT_Type->addItem(QString("%1").arg(bt_type_name[ii]));
    }
    ui->comboBox_BT_Type->setCurrentIndex(0);
    // [GBT]
    for(int ii=0; ii<gbt_type_num; ii++) {
        ui->comboBox_GBT_Type->addItem(QString("%1").arg(gbt_type_name[ii]));
    }
    ui->comboBox_GBT_Type->setCurrentIndex(0);
    // [RF]
    for(int ii=0; ii<rf_termcrit_num; ii++) {
        ui->comboBox_RF_TermcritType->addItem(QString("%1").arg(rf_termcrit_name[ii]));
    }
    ui->comboBox_RF_TermcritType->setCurrentIndex(0);
    // [ERT]
    for(int ii=0; ii<rf_termcrit_num; ii++) {
        ui->comboBox_ERT_TermcritType->addItem(QString("%1").arg(rf_termcrit_name[ii]));
    }
    ui->comboBox_ERT_TermcritType->setCurrentIndex(0);
    // [ANN]
    ui->spinBox_ANN_SizeLayer2->setEnabled(ui->checkBox_ANN_SizeLayer2->isChecked());
    ui->spinBox_ANN_SizeLayer3->setEnabled(ui->checkBox_ANN_SizeLayer3->isChecked());
    for(int ii=0; ii<ann_method_num; ii++) {
        ui->comboBox_ANN_Method->addItem(QString("%1").arg(ann_method_name[ii]));
    }
    ui->comboBox_ANN_Method->setCurrentIndex(0);
    for(int ii=0; ii<ann_fx_num; ii++) {
        ui->comboBox_ANN_Fx->addItem(QString("%1").arg(ann_fx_name[ii]));
    }
    ui->comboBox_ANN_Fx->setCurrentIndex(0);
    for(int ii=0; ii<rf_termcrit_num; ii++) {
        ui->comboBox_ANN_Termcrit_Type->addItem(QString("%1").arg(rf_termcrit_name[ii]));
    }
    ui->comboBox_ANN_Termcrit_Type->setCurrentIndex(0);
    // [EM]
    for(int ii=0; ii<em_covmatrixtype_num; ii++) {
        ui->comboBox_EM_CoxMatrixType->addItem(QString("%1").arg(em_covmatrixtype_name[ii]));
    }
    ui->comboBox_EM_CoxMatrixType->setCurrentIndex(0);
    for(int ii=0; ii<rf_termcrit_num; ii++) {
        ui->comboBox_EM_Termcrit_Type->addItem(QString("%1").arg(rf_termcrit_name[ii]));
    }
    ui->comboBox_EM_Termcrit_Type->setCurrentIndex(0);
    // [EMC]
    for(int ii=0; ii<em_covmatrixtype_num; ii++) {
        ui->comboBox_EMC_CoxMatrixType->addItem(QString("%1").arg(em_covmatrixtype_name[ii]));
    }
    ui->comboBox_EMC_CoxMatrixType->setCurrentIndex(0);
    for(int ii=0; ii<rf_termcrit_num; ii++) {
        ui->comboBox_EMC_Termcrit_Type->addItem(QString("%1").arg(rf_termcrit_name[ii]));
    }
    ui->comboBox_EMC_Termcrit_Type->setCurrentIndex(0);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::goToNextClassIdx()
{
    currentClassIdx++;
    if(currentClassIdx>=getNumberOfClasses()) {
        currentClassIdx = 0;
    }
    ui->comboBoxCurrentClassIdx->setCurrentIndex(currentClassIdx);
}

void MainWindow::goToPrevClassIdx()
{
    currentClassIdx--;
    if(currentClassIdx<0) {
        currentClassIdx = getNumberOfClasses()-1;
    }
    ui->comboBoxCurrentClassIdx->setCurrentIndex(currentClassIdx);
}

int MainWindow::getNumberOfClasses() const
{
    return ui->spinBoxNumberOfClasses->value();
}

int MainWindow::getClusterSize() const
{
    return ui->spinBoxClusterSize->value();
}

int MainWindow::getClusterRadius() const
{
    return ui->spinBoxClusterRadius->value();
}

bool MainWindow::isDrawMode() const
{
    return ui->checkBoxIsDrawMode->isChecked();
}

void MainWindow::setNumberOfClasses(int numberOfClasses)
{
    int oldNumberOfClasses  = ui->comboBoxCurrentClassIdx->count();
    if(numberOfClasses<oldNumberOfClasses) {
        ui->label->refreshValidClasses();
        ui->label->update();
    }
    int oldIdx  = ui->comboBoxCurrentClassIdx->currentIndex();
    if(oldIdx<0) {
        oldIdx  = 0;
    }
    ui->comboBoxCurrentClassIdx->clear();
    for(int ii=0; ii<numberOfClasses; ii++) {
        ui->comboBoxCurrentClassIdx->addItem(listClassIcons.at(ii), QString("class #%1").arg(ii));
    }
    if(oldIdx<ui->comboBoxCurrentClassIdx->count()) {
        ui->comboBoxCurrentClassIdx->setCurrentIndex(oldIdx);
    } else {
        ui->comboBoxCurrentClassIdx->setCurrentIndex(numberOfClasses-1);
    }
//    qDebug() << ui->comboBoxCurrentClassIdx->currentIndex();
    setCurrentClassIdx(ui->comboBoxCurrentClassIdx->currentIndex());
}

void MainWindow::on_spinBoxNumberOfClasses_valueChanged(int num)
{
    setNumberOfClasses(num);
}

void MainWindow::setCurrentClassIdx(int currentClassIdx)
{
    this->currentClassIdx   = currentClassIdx;
}

void MainWindow::buildListClassColors()
{
    listClassIcons.clear();
    listClassColors.clear();
    for(int ii=0; ii<DEF_NUMCLASS_MAX+20; ii++) {
        QPixmap pxm(128,128);
        QColor c(qrand()%255,qrand()%255,qrand()%255);
        listClassColors.append(c);
        pxm.fill(c);
        listClassIcons.append(QIcon(pxm));
    }
}

void MainWindow::on_pushButtonRefreshClassColors_clicked()
{
    buildListClassColors();
    setNumberOfClasses(ui->spinBoxNumberOfClasses->value());
    ui->label->update();
}

void MainWindow::on_comboBoxCurrentClassIdx_currentIndexChanged(int index)
{
    setCurrentClassIdx(index);
}

void MainWindow::on_pushButtonClearPoints_clicked()
{
    ui->label->clearPoints();
}

void MainWindow::on_pushButtonApplyClassifier_clicked()
{
    qDebug() << ui->toolBox->currentIndex();
    uint clsIdx  = ui->toolBox->currentIndex();
    if(clsIdx>=listOfClassifiers.size()) {
        QMessageBox::information(this, "ERROR", "this classifier is not realised!");
        return;
    }
    ClassifierInterface* clsPtr = listOfClassifiers[clsIdx];
    if(!clsPtr->isTrained()) {
        QMessageBox::information(this, "ERROR", "Need train classifier!");
        return;
    }
    QPixmap pxm(width(), height());
    QPainter pnt(&pxm);
    ui->progressBar->setMinimum(0);
    ui->progressBar->setMinimum(width());
    ui->progressBar->setValue(0);
    int r1 = DEF_POINT_RADIUS;
    int r2 = DEF_POINT_RADIUS*2;
    QBrush brush(Qt::SolidPattern);
    for(int xx=r1; xx<width()-r1; xx+=r2) {
        for(int yy=r1; yy<height()-r1; yy+=r2) {
            brush.setColor(listClassColors.at(clsPtr->classify(xx,yy)));
            pnt.setBrush(brush);
            pnt.drawRect(xx-r1, yy-r1, r2, r2);
        }
    }
    ui->labelStatusInfo->setText(QString("Error self-calss: %1 %").arg(100.*clsPtr->calcSelfError()));
    ui->label->setPixmap(pxm);
    ui->label->update();
}

void MainWindow::on_pushButtonTrainClassifier_clicked()
{
    uint clsIdx  = ui->toolBox->currentIndex();
    if(clsIdx>=listOfClassifiers.size()) {
        return;
    }
    ClassifierInterface* clsPtr = listOfClassifiers[clsIdx];
    int maxClassNum = clsPtr->getMaxClassNum();
    if(getNumberOfClasses()>maxClassNum) {
        ui->spinBoxNumberOfClasses->setValue(maxClassNum);
        QMessageBox::information(this, "WARNING", QString("This classifier type require max #classes = %1, extra classes will be deleted.").arg(maxClassNum));
        setNumberOfClasses(maxClassNum);
    }
    ui->tabWidget->setEnabled(false);
    ui->labelStatusInfo->setText("wait...");
    QApplication::processEvents();
    switch (clsIdx) {
    case CLSF_IDX_NBC:
        clsPtr->trainData(ui->label->listPointsOCV, ui->label->listLabelsOCV);
        break;
    case CLSF_IDX_KNN:
    {
        ClassifierKNN* tmp  = static_cast<ClassifierKNN*>(clsPtr);
        tmp->setNumKNN(ui->spinBoxKNN_knn->value());
        tmp->trainData(ui->label->listPointsOCV, ui->label->listLabelsOCV);
    }
        break;
    case CLSF_IDX_SVM:
    {
        ClassifierSVM* tmp  = static_cast<ClassifierSVM*>(clsPtr);
        tmp->setParameters(
                svm_type_idx[ui->comboBox_SVM_Type->currentIndex()],
                svm_kernel_idx[ui->comboBox_SVM_Kernel->currentIndex()],
                ui->doubleSpinBox_SVM_Degree->value(),
                ui->doubleSpinBox_SVM_Gamma->value(),
                ui->doubleSpinBox_SVM_Coef0->value(),
                ui->doubleSpinBox_SVM_C->value(),
                ui->doubleSpinBox_SVM_Nu->value());
        tmp->trainData(ui->label->listPointsOCV, ui->label->listLabelsOCV);
    }
        break;
    case CLSF_IDX_DT:
    {
        ClassifierDT* tmp   = static_cast<ClassifierDT*>(clsPtr);
        tmp->setParameters(
                    ui->spinBox_DT_MaxDepth->value(),
                    ui->spinBox_DT_MinSampleCount->value(),
                    ui->checkBox_DT_UseSurrogates->isChecked(),
                    ui->checkBox_DT_Use1seRule->isChecked(),
                    ui->checkBox_DT_TruncatePrunedTree->isChecked(),
                    ui->spinBox_DT_NumCrossvalidationFolds->value(),
                    ui->doubleSpinBox_DT_RegressionAccuracy->value());
        tmp->trainData(ui->label->listPointsOCV, ui->label->listLabelsOCV);
    }
        break;
    case CLSF_IDX_BT:
    {
        ClassifierBT* tmp   = static_cast<ClassifierBT*>(clsPtr);
        tmp->setParameters(
                bt_type_idx[ui->comboBox_BT_Type->currentIndex()],
                ui->spinBox_BT_WeakCount->value(),
                ui->doubleSpinBox_BT_WeightTrimRate->value(),
                ui->spinBox_BT_MaxDepth->value(),
                ui->checkBox_BT_UseSurrogates->isChecked());
        tmp->trainData(ui->label->listPointsOCV, ui->label->listLabelsOCV);
    }
        break;
    case CLSF_IDX_GBT:
    {
        ClassifierGBT* tmp   = static_cast<ClassifierGBT*>(clsPtr);
        tmp->setParameters(
                gbt_type_idx[ui->comboBox_GBT_Type->currentIndex()],
                ui->spinBox_GBT_WeakCount->value(),
                ui->doubleSpinBox_GBT_Shrinkage->value(),
                ui->doubleSpinBox_GBT_SubsamplePortion->value(),
                ui->spinBox_GBT_MaxDepth->value(),
                ui->checkBox_GBT_UseSurrogates->isChecked());
        tmp->trainData(ui->label->listPointsOCV, ui->label->listLabelsOCV);
    }
        break;
    case CLSF_IDX_RF:
    {
        ClassifierRF* tmp   = static_cast<ClassifierRF*>(clsPtr);
        tmp->setParameters(
                    ui->spinBox_RF_MaxDepth->value(),
                    ui->spinBox_RF_MinSampleCount->value(),
                    ui->doubleSpinBox_RF_RegressionAccuracy->value(),
                    ui->checkBox_RF_UseSurrogates->isChecked(),
                    ui->spinBox_RF_MaxCategories->value(),
                    ui->checkBox_RF_CalcVarImportance->isChecked(),
                    ui->spinBox_RF_NativeVars->value(),
                    ui->spinBox_RF_MaxNumTreesInForest->value(),
                    ui->doubleSpinBox_RF_ForestAccuracy->value(),
                    rf_termcrit_idx[ui->comboBox_RF_TermcritType->currentIndex()]);
        tmp->trainData(ui->label->listPointsOCV, ui->label->listLabelsOCV);
    }
        break;
    case CLSF_IDX_ERT:
    {
        ClassifierERT* tmp   = static_cast<ClassifierERT*>(clsPtr);
        tmp->setParameters(
                    ui->spinBox_ERT_MaxDepth->value(),
                    ui->spinBox_ERT_MinSampleCount->value(),
                    ui->doubleSpinBox_ERT_RegressionAccuracy->value(),
                    ui->checkBox_ERT_UseSurrogates->isChecked(),
                    ui->spinBox_ERT_MaxCategories->value(),
                    ui->checkBox_ERT_CalcVarImportance->isChecked(),
                    ui->spinBox_ERT_NativeVars->value(),
                    ui->spinBox_ERT_MaxNumTreesInForest->value(),
                    ui->doubleSpinBox_ERT_ForestAccuracy->value(),
                    rf_termcrit_idx[ui->comboBox_ERT_TermcritType->currentIndex()]);
        tmp->trainData(ui->label->listPointsOCV, ui->label->listLabelsOCV);
    }
        break;
    case CLSF_IDX_ANN:
    {
        ClassifierANN* tmp  = static_cast<ClassifierANN*>(clsPtr);
        tmp->setParameters(getNumberOfClasses(),
                           ann_method_idx[ui->comboBox_ANN_Method->currentIndex()],
                           ann_fx_idx[ui->comboBox_ANN_Fx->currentIndex()],
                           ui->doubleSpinBox_ANN_Param1->value(),
                           ui->doubleSpinBox_ANN_Param2->value(),
                           ui->spinBox_ANN_SizeLayer1->value(),
                           ui->checkBox_ANN_SizeLayer2->isChecked()?ui->spinBox_ANN_SizeLayer2->value():-1,
                           ui->checkBox_ANN_SizeLayer3->isChecked()?ui->spinBox_ANN_SizeLayer3->value():-1,
                           rf_termcrit_idx[ui->comboBox_ANN_Termcrit_Type->currentIndex()],
                           ui->spinBox_ANN_Termcrit_NumIter->value(),
                           ui->doubleSpinBox_ANN_Termcrit_Accuracy->value());
        tmp->trainData(ui->label->listPointsOCV, ui->label->listLabelsOCV);
    }
        break;
    case CLSF_IDX_EM:
    {
        ClassifierEM* tmp  = static_cast<ClassifierEM*>(clsPtr);
        tmp->setParameters(ui->spinBox_EM_NumClusters->value(),
                           em_covmatrixtype_idx[ui->comboBox_EM_CoxMatrixType->currentIndex()],
                           rf_termcrit_idx[ui->comboBox_EM_Termcrit_Type->currentIndex()],
                           ui->spinBox_EM_Termcrit_NumIter->value(),
                           ui->doubleSpinBox_EM_Termcrit_Accuracy->value());
        tmp->trainData(ui->label->listPointsOCV, ui->label->listLabelsOCV);
    }
        break;
    case CLSF_IDX_EMC:
    {
        ClassifierEMC* tmp  = static_cast<ClassifierEMC*>(clsPtr);
        tmp->setParameters(getNumberOfClasses(),
                           ui->spinBox_EMC_NumClusters->value(),
                           em_covmatrixtype_idx[ui->comboBox_EMC_CoxMatrixType->currentIndex()],
                           rf_termcrit_idx[ui->comboBox_EMC_Termcrit_Type->currentIndex()],
                           ui->spinBox_EMC_Termcrit_NumIter->value(),
                           ui->doubleSpinBox_EMC_Termcrit_Accuracy->value());
        tmp->trainData(ui->label->listPointsOCV, ui->label->listLabelsOCV);
    }
        break;
    default:
            break;
    }
    ui->tabWidget->setEnabled(true);
    ui->labelStatusInfo->setText("...");
    QApplication::processEvents();
    clsPtr->loadAdditionalPoints(ui->label->listExtPoints);
    qDebug() << "classifier(" << clsPtr->toQString() << ")";
}

void MainWindow::on_checkBoxIsDrawMode_clicked()
{
    if(isDrawMode()) {
        ui->pushButtonTrainClassifier->setEnabled(false);
        ui->pushButtonApplyClassifier->setEnabled(false);
    } else {
        ui->pushButtonTrainClassifier->setEnabled(true);
        ui->pushButtonApplyClassifier->setEnabled(true);
    }
    ui->label->update();
}

void MainWindow::on_action_Exit_triggered()
{
    qApp->quit();
}

void MainWindow::on_actionFullscreen_triggered()
{
    if(!isFullScreen()) {
        showFullScreen();
    } else {
        showNormal();
    }
}

void MainWindow::on_toolButton_clicked()
{
    uint clsIdx  = ui->toolBox->currentIndex();
    switch (clsIdx) {
    case CLSF_IDX_NBC:
        QDesktopServices::openUrl(QUrl("http://docs.opencv.org/modules/ml/doc/normal_bayes_classifier.html", QUrl::TolerantMode));
        break;
    case CLSF_IDX_KNN:
        QDesktopServices::openUrl(QUrl("http://docs.opencv.org/modules/ml/doc/k_nearest_neighbors.html", QUrl::TolerantMode));
        break;
    case CLSF_IDX_SVM:
        QDesktopServices::openUrl(QUrl("http://docs.opencv.org/modules/ml/doc/support_vector_machines.html", QUrl::TolerantMode));
        break;
    case CLSF_IDX_DT:
        QDesktopServices::openUrl(QUrl("http://docs.opencv.org/modules/ml/doc/decision_trees.html", QUrl::TolerantMode));
        break;
    case CLSF_IDX_BT:
        QDesktopServices::openUrl(QUrl("http://docs.opencv.org/modules/ml/doc/boosting.html", QUrl::TolerantMode));
        break;
    case CLSF_IDX_GBT:
        QDesktopServices::openUrl(QUrl("http://docs.opencv.org/modules/ml/doc/gradient_boosted_trees.html", QUrl::TolerantMode));
        break;
    case CLSF_IDX_RF:
        QDesktopServices::openUrl(QUrl("http://docs.opencv.org/modules/ml/doc/random_trees.html", QUrl::TolerantMode));
        break;
    case CLSF_IDX_ERT:
        QDesktopServices::openUrl(QUrl("http://docs.opencv.org/modules/ml/doc/ertrees.html", QUrl::TolerantMode));
        break;
    case CLSF_IDX_ANN:
        QDesktopServices::openUrl(QUrl("http://docs.opencv.org/modules/ml/doc/neural_networks.html", QUrl::TolerantMode));
        break;
    case CLSF_IDX_EM:
        QDesktopServices::openUrl(QUrl("http://docs.opencv.org/modules/ml/doc/expectation_maximization.html", QUrl::TolerantMode));
        break;
    case CLSF_IDX_EMC:
        QDesktopServices::openUrl(QUrl("http://docs.opencv.org/modules/ml/doc/expectation_maximization.html", QUrl::TolerantMode));
        break;
    default:
        break;
    }
}

void MainWindow::on_checkBox_ANN_SizeLayer2_clicked()
{
    if(!ui->checkBox_ANN_SizeLayer2->isChecked()) {
        ui->spinBox_ANN_SizeLayer2->setEnabled(false);
        ui->checkBox_ANN_SizeLayer3->setChecked(false);
        ui->spinBox_ANN_SizeLayer3->setEnabled(false);
    } else {
        ui->spinBox_ANN_SizeLayer2->setEnabled(true);
    }
}

void MainWindow::on_checkBox_ANN_SizeLayer3_clicked()
{
    if(ui->checkBox_ANN_SizeLayer3->isChecked()) {
        ui->spinBox_ANN_SizeLayer3->setEnabled(true);
        ui->checkBox_ANN_SizeLayer2->setChecked(true);
        ui->spinBox_ANN_SizeLayer2->setEnabled(true);
    } else {
        ui->spinBox_ANN_SizeLayer3->setEnabled(false);
    }
}
