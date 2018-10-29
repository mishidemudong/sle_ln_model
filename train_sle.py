# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 18:35:41 2018

@author: ldk
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adagrad,SGD
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import simpleRNNModel,DLModel
import auc_ks_eval
import readFunc
from keras.utils import np_utils
 

def train(modelfile,model,train_x, train_y,x_val,y_val):
    #iteration times
    epoch_1 = 120

    k = 1
    batch_size_1 = 48*k

    #binary task
    opt1 = SGD(lr=0.075, decay=1e-6)
    #multilable tasks
#    opt3 = Adam(lr=0.001, decay=1e-6)
#        model.compile(loss='binary_crossentropy', optimizer=opt1 ,metrics=['binary_accuracy'])  ##auc,,loss:categorical_crossentropy  sparse_categorical_crossentropy,
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=opt1,
                  loss= 'mse',  #mse,binary_crossentropy
                  metrics=['accuracy'])
    
    #save best model
    checkpoint = ModelCheckpoint(modelfile, monitor='val_acc', verbose=1, save_best_only=True,
                            mode='max')
    callbacks_list = [checkpoint]
    
    
    #to solve samples not 
    sw = {0: 2, 1: 10}
    
    model.fit(train_x, train_y, epochs=epoch_1, batch_size=batch_size_1,shuffle = False,
              callbacks=callbacks_list,
              validation_data=(x_val,y_val),
              class_weight=sw,
              verbose=1)      
    
    return model


def train_rnn(modelfile,model,train_x, train_y,x_val,y_val,MAX_SEQUENCE_LENGTH):
    #iteration times
    epoch_1 = 1

    k = 1
    batch_size_1 = 48*k

    #binary task
    opt1 = SGD(lr=0.075, decay=1e-6)
    model.compile(optimizer=opt1,
                  loss= 'mse',  #mse,binary_crossentropy
                  metrics=['accuracy'])
    
    #save best model
    checkpoint = ModelCheckpoint(modelfile, monitor='val_acc', verbose=1, save_best_only=True,
                            mode='max')
    callbacks_list = [checkpoint]
    
    #to solve samples not 
    sw = {0: 2, 1: 10}
    
    #生成适合模型输入的格式
    
#    d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(maxlen-len(x))))
#    d['y'] = d['label'].apply(trans_one)
    print("data done!")
    
    model.fit(train_x, train_y, epochs=epoch_1, batch_size=batch_size_1,shuffle = False,
              callbacks=callbacks_list,
              validation_data=(x_val,y_val),
              class_weight=sw,
              verbose=1) #4训练模型        
    
    return model

def train_BERT(modelfile,model,train_x, train_y,x_val,y_val,MAX_SEQUENCE_LENGTH):
    from keras.optimizers import Adagrad,SGD
    from keras.callbacks import ModelCheckpoint
    #iteration times
    epoch_1 = 10

    k = 1
    batch_size_1 = 48*k

    #binary task
    opt1 = SGD(lr=0.075, decay=1e-6)
#    opt1 = Adagrad(lr=0.075, decay=1e-6)
    model.compile(optimizer=opt1,
                  loss= 'mse',  #mse,binary_crossentropy
                  metrics=['accuracy'])
    
    #save best model
    checkpoint = ModelCheckpoint(modelfile, monitor='val_acc', verbose=1, save_best_only=True,
                            mode='max')
    callbacks_list = [checkpoint]
    
    #to solve samples not 
    sw = {0: 2, 1: 10}
    
    #生成适合模型输入的格式
    
#    d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(maxlen-len(x))))
#    d['y'] = d['label'].apply(trans_one)
    print("data done!")
    
    model.fit(train_x, train_y, epochs=epoch_1, batch_size=batch_size_1,shuffle = False,
              callbacks=callbacks_list,
              validation_data=(x_val,y_val),
              class_weight=sw,
              verbose=1) #4训练模型        
    
    return model

if __name__ == '__main__':
    modelfile = './model/bert/rnnmodel_20181229_agd.h5' #srnnmodel_20181229_l2
    train_file = './data/train_data/result_LN0-1_2018-10-27.csv'
    VAC_DIR = './data/vac/sle_LN0-1_20181027.vector.txt'  #sle_LN0-2_20181024.vector,sle_1024.vector
    MAX_SEQUENCE_LENGTH = 24*30
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.66)  
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  

#    #dnn model
#    x_train,y_train,x_val,y_val, num_words, embedding_matrix = readFunc.read_data(train_file,VAC_DIR,MAX_SEQUENCE_LENGTH) #,num_words,embedding_matrix
##    onehottrain_y = tf.one_hot(np.asarray(y_train), 2).eval()
##    print ("onehottrain_y",y_train[:1])
#  
#    wrnnmodel = simpleRNNModel.DNNModel().buildnet(x_train,2, num_words, embedding_matrix)
#    train(modelfile,wrnnmodel,x_train, y_train,x_val,y_val)
#    predict = wrnnmodel.predict_classes(x_val)
#    print (np.argmax(y_val,axis=1)[:10])
#    print (predict[:10])
#    
#    wrnnmodel.save(modelfile)
#    
#    auc_ks_eval.model_evaluation(wrnnmodel,x_train, np.argmax(y_train,axis=1), x_val, np.argmax(y_val,axis=1))

#    
    
    #rnn model
#    x_train,y_train,x_val,y_val, num_words, embedding_matrix = readFunc.read_data(train_file,VAC_DIR,MAX_SEQUENCE_LENGTH) #,num_words,embedding_matrix
#    wrnnmodel = simpleRNNModel.WRNNModel().buildnet(x_train,2, num_words, embedding_matrix)
#    wrnnmodel = train_rnn(modelfile,wrnnmodel,x_train, y_train,x_val,y_val,MAX_SEQUENCE_LENGTH)
#    
#    predict = wrnnmodel.predict(x_val)
#    print (np.argmax(y_val,axis=1)[:10])
#    print (predict[:10])
#    
#    wrnnmodel.save(modelfile)
#    auc_ks_eval.model_evaluation(wrnnmodel,x_train, np.argmax(y_train,axis=1), x_val, np.argmax(y_val,axis=1))
    
    #BERT
    x_train,y_train,x_val,y_val, num_words, embedding_matrix = readFunc.read_data(train_file,VAC_DIR,MAX_SEQUENCE_LENGTH) #,num_words,embedding_matrix
    wrnnmodel = DLModel.BERTDNNModel().buildnet(x_train,2, num_words, embedding_matrix)
    wrnnmodel = train_BERT(modelfile,wrnnmodel,x_train, y_train,x_val,y_val,MAX_SEQUENCE_LENGTH)
    
    predict = wrnnmodel.predict(x_val)
    print (np.argmax(y_val,axis=1)[:10])
    print (predict[:10])
    
    wrnnmodel.save(modelfile)
    auc_ks_eval.model_evaluation(wrnnmodel,x_train, np.argmax(y_train,axis=1), x_val, np.argmax(y_val,axis=1))
    
    sess.close()    
   
