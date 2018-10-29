# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 23:25:33 2018

@author: ldk
"""

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
import sklearn.metrics as metrics
import pandas as pd 
import numpy as np

def ks_func(train_y,dtrain_predictions,test_y,dtest_predictions):
    fpr_train, tpr_train, threshold = metrics.roc_curve(train_y, dtrain_predictions, pos_label=1)
    ks_train = max(abs(fpr_train - tpr_train))
    fpr_test, tpr_test, _ = metrics.roc_curve(test_y, dtest_predictions, pos_label=1)
    ks_test = max(abs(fpr_test - tpr_test))
    
    return ks_train,ks_test


def seqential_evaluation(mlp_model,train_x, train_y, test_x, test_y):
    #对训练集预测
    train_result = pd.value_counts(train_y)
    train_total = len(train_y)
    train_posi_c = train_result[1]
    train_nega_c = train_result[0]
    
    train_posi_r = train_posi_c / train_total
    train_nega_r = train_nega_c / train_total
    
    test_result = pd.value_counts(test_y)
    test_total = len(test_y)
    test_posi_c = test_result[1]
    test_nega_c = test_result[0]
    
    test_posi_r = test_posi_c / test_total
    test_nega_r = test_nega_c / test_total    
    
    
    
    dtrain_predictions = mlp_model.predict_classes(train_x)
    dtrain_predprob = mlp_model.predict_proba(train_x)[:, 1]
    dtest_predictions = mlp_model.predict_classes(test_x)
    dtest_predprob = mlp_model.predict_proba(test_x)[:, 1]

    train_f1_score = f1_score(train_y, dtrain_predictions)
    test_f1_score = f1_score(test_y, dtest_predictions)

    # 输出模型的一些结果
    
    print('-------训练集--------->')
    print("总数:%d \n正样本:%d \n负样本:%d \n正样本占比:%.2f%% \n负样本占比:%.2f%%"%(train_total,train_posi_c,train_nega_c,train_posi_r*100,train_nega_r*100))
    print("准确率 : %.4g" % accuracy_score(train_y, dtrain_predictions))
    print("AUC 得分 (训练集): %f" % roc_auc_score(train_y, dtrain_predprob))
    print("f1_score (训练集): %f" % train_f1_score)
    print(confusion_matrix(train_y, dtrain_predictions))
    
    
    # 输出模型的一些结果
    print('-------测试集--------->')
    print("总数:%d \n正样本:%d \n负样本:%d \n正样本占比:%.2f%% \n负样本占比:%.2f%%"%(test_total,test_posi_c,test_nega_c,test_posi_r*100,test_nega_r*100))
    print("准确率 : %.4g" % accuracy_score(test_y, dtest_predictions))
    print("AUC 得分 (测试集): %f" % roc_auc_score(test_y, dtest_predprob))
    print("f1_score (测试集): %f" % test_f1_score)
    print(confusion_matrix(test_y, dtest_predictions))
    
    print ("max ks 训练集：%.5f,测试集：%.5f"%ks_func(train_y,dtrain_predictions,test_y,dtest_predictions))
    

def model_evaluation(mlp_model,train_x, train_y, test_x, test_y):
    #对训练集预测
    train_result = pd.value_counts(train_y)
    train_total = len(train_y)

    train_posi_c = train_result[1]
    train_nega_c = train_result[0]
    
    train_posi_r = train_posi_c / train_total
    train_nega_r = train_nega_c / train_total
    
    test_result = pd.value_counts(test_y)
    test_total = len(test_y)
    test_posi_c = test_result[1]
    test_nega_c = test_result[0]
    
    test_posi_r = test_posi_c / test_total
    test_nega_r = test_nega_c / test_total    
    
    
    dtrain_predictions = np.argmax(mlp_model.predict(train_x),axis=1)
    dtrain_predprob = mlp_model.predict(train_x)[:, 1]
    dtest_predictions = np.argmax(mlp_model.predict(test_x),axis=1)
    dtest_predprob = mlp_model.predict(test_x)[:, 1]

    train_f1_score = f1_score(train_y, dtrain_predictions)
    test_f1_score = f1_score(test_y, dtest_predictions)

    # 输出模型的一些结果
    
    print('-------训练集--------->')
    print("总数:%d \n正样本:%d \n负样本:%d \n正样本占比:%.2f%% \n负样本占比:%.2f%%"%(train_total,train_posi_c,train_nega_c,train_posi_r*100,train_nega_r*100))
    print("准确率 : %.4g" % accuracy_score(train_y, dtrain_predictions))
    print("AUC 得分 (训练集): %f" % roc_auc_score(train_y, dtrain_predprob))
    print("f1_score (训练集): %f" % train_f1_score)
    print(confusion_matrix(train_y, dtrain_predictions))
    
    
    # 输出模型的一些结果
    print('-------测试集--------->')
    print("总数:%d \n正样本:%d \n负样本:%d \n正样本占比:%.2f%% \n负样本占比:%.2f%%"%(test_total,test_posi_c,test_nega_c,test_posi_r*100,test_nega_r*100))
    print("准确率 : %.4g" % accuracy_score(test_y, dtest_predictions))
    print("AUC 得分 (测试集): %f" % roc_auc_score(test_y, dtest_predprob))
    print("f1_score (测试集): %f" % test_f1_score)
    print(confusion_matrix(test_y, dtest_predictions))
    
    print ("max ks 训练集：%.5f,测试集：%.5f"%ks_func(train_y,dtrain_predictions,test_y,dtest_predictions))