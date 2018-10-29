# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:38:32 2018

@author: ldk
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd

def read_data(train_file,VAC_DIR,MAX_SEQUENCE_LENGTH):

#    MAX_SEQUENCE_LENGTH = 1200  # 每个文本或者句子的截断长度，只保留1000个单词
    MAX_NUM_WORDS = 20000  # 用于构建词向量的词汇表数量
    EMBEDDING_DIM = 100  # 词向量维度
    VALIDATION_SPLIT = 0.3
     
    
    # 构建词向量索引
    print("Indexing word vectors.")
    embeddings_index = {}
    with open(VAC_DIR, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]  # 单词
            coefs = np.asarray(values[1:], dtype='float32')  # 单词对应的向量
            embeddings_index[word] = coefs  # 单词及对应的向量
     
    print('Found %s word vectors.'%len(embeddings_index))#400000个单词和词向量
     
    print('预处理文本数据集')
    texts = []  # 训练文本样本的list
    labels = []  # 标签list
     
    #读取训练数
    data = pd.read_csv(train_file)
    
    texts = data['Item'].tolist()
    labels = data['Tag'].replace('non-LN',0).replace('LN',1).tolist()
     
    print("Found %s texts %s label_id." % (len(texts), len(labels)))  # 19997个文本文件
     
    # 向量化文本样本
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    # fit_on_text(texts) 使用一系列文档来生成token词典，texts为list类，每个元素为一个文档。就是对文本单词进行去重后
    tokenizer.fit_on_texts(texts)
    # texts_to_sequences(texts) 将多个文档转换为word在词典中索引的向量形式,shape为[len(texts)，len(text)] -- (文档数，每条文档的长度)
    sequences = tokenizer.texts_to_sequences(texts)
    print(sequences[0])
    print(len(sequences))  # 19997
     
    word_index = tokenizer.word_index  # word_index 一个dict，保存所有word对应的编号id，从1开始
    print("Founnd %s unique tokens." % len(word_index))  # 174074个单词
    # ['the', 'to', 'of', 'a', 'and', 'in', 'i', 'is', 'that', "'ax"] [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(list(word_index.keys())[0:10], list(word_index.values())[0:10])  #
     
    
    ######非常好用的函数，直接用
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # 长度超过MAX_SEQUENCE_LENGTH则截断，不足则补0
     
    labels = to_categorical(np.asarray(labels))
    print("训练数据大小为：", data.shape)  # (19997, 1000)
    print("标签大小为:", labels.shape)  # (19997, 20)
     
    # 将训练数据划分为训练集和验证集
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)  # 打乱数据
    data = data[indices]
    labels = labels[indices]
     
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
     
    # 训练数据
    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
     
    # 验证数据
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
     
    # 准备词向量矩阵
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)  # 词汇表数量
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))  # 20000*100
     
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:  # 过滤掉根据频数排序后排20000以后的词
            continue
        embedding_vector = embeddings_index.get(word)  # 根据词向量字典获取该单词对应的词向量
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    return x_train,y_train,x_val,y_val,num_words,embedding_matrix