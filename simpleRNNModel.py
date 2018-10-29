# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:23:00 2018

@author: ldk
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,GaussianDropout,Embedding,Flatten,LSTM,SimpleRNN,Bidirectional,TimeDistributed
from tensorflow.keras.layers import Dropout,BatchNormalization
from tensorflow.keras.constraints import MaxNorm as maxnorm  
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import Input
from tensorflow.keras.models import Model

class WRNNModel:
    def Average(self,inputs_list,weight_array):
        """Layer that averages a list of inputs.
    
        It takes as input a list of tensors,
        all of the same shape, and returns
        a single tensor (also of the same shape).
        """
        output = inputs_list[0] * weight_array[0]
        for i in range(1, len(inputs_list)):
            output += inputs_list[i] * weight_array[i]
        return output
            
    
    def buildnet(self,trainX, output_d, num_words, embedding_matrix):
        #prepare the embedding layer
        #for same type medical events         

#        #Lstm import hyperparameters 
#        lstm_hid_size = 128
#        l2_rate = 0.000000000001
        
        sequence = Input(shape=(trainX.shape[1],), dtype='int32')
        
        embedded = Embedding(128, 80, input_length=trainX.shape[1], mask_zero=True)(sequence)
        
        blstm = Bidirectional(LSTM(32, return_sequences=True), merge_mode='sum')(embedded)
        blstm = Bidirectional(LSTM(32))(blstm)
        
#        output = TimeDistributed(Dense(units = output_d, input_dim=64, use_bias=False,
#                                       activation='sigmoid', 
#                                       name='Model_loss'))(blstm)
        
        output = Dense(units = output_d, input_dim=32, use_bias=False,
                               activation='sigmoid', 
                               name='Model_loss')(blstm)
                
        print("output",output)
        
        Main_model = Model(inputs=sequence, outputs=output)   
        
        return Main_model


class DNNModel:
    def Average(self,inputs_list,weight_array):
        """Layer that averages a list of inputs.
    
        It takes as input a list of tensors,
        all of the same shape, and returns
        a single tensor (also of the same shape).
        """
        output = inputs_list[0] * weight_array[0]
        for i in range(1, len(inputs_list)):
            output += inputs_list[i] * weight_array[i]
        return output
            
    
    def buildnet(self,trainX, output_d, num_words, embedding_matrix):
        #prepare the embedding layer
        #for same type medical events         
        # 准备词向量矩阵
        
#        sequence_input = Input(shape=(None,1000), dtype='int32')
#        
#        # 加载预训练的词向量到Embedding layer
#        RNN_emb =  Embedding(input_dim=num_words,  # 词汇表单词数量
#                                    output_dim=100,  # 词向量维度
#                                    weights=[embedding_matrix],
#                                    input_length=2000)  # 词向量矩阵不进行训练
#        RNN_emb = embedding_layer(sequence_input)
        #other type medical events
        RNN_emb = Embedding(128, 120,input_length=trainX.shape[1])#,embeddings_regularizer=regularizers.l2(0.000045)
        
        #build model        
        #define main_model input
                
        Main_model = Sequential()        

        Main_model.add(RNN_emb) #
        print ("1layer Embedding shape",Main_model.output_shape)
        
        Main_model.add(Flatten())
        
        ##add attention layer implement with keras
#        Main_model.add(Attention_keras())
        
        #Lstm import hyperparameters 
        lstm_hid_size = 128
        l2_rate = 0.05
        
        #dense layer
        Main_model.add(Dense(units=lstm_hid_size, input_dim=lstm_hid_size,kernel_initializer='glorot_uniform',activation='relu',use_bias=False,
                        kernel_regularizer=regularizers.l2(l2_rate)))

        Main_model.add(Dense(units=lstm_hid_size, input_dim=lstm_hid_size,kernel_initializer='glorot_uniform',activation='relu',use_bias=False,
                        kernel_regularizer=regularizers.l2(l2_rate)))

        Main_model.add(Dense(units=lstm_hid_size, input_dim=lstm_hid_size,kernel_initializer='glorot_uniform',activation='relu',use_bias=False)) 
        
        Main_model.add(Dropout(0.2))

        Main_model.add(Dense(units = output_d, input_dim=64, use_bias=False,
                        activation='sigmoid', 
                        name='Model_loss'))#W_constraint=maxnorm(1),
        

        
        return Main_model
