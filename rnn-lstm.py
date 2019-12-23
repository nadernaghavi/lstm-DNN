# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 01:58:46 2019

@author: NaderBrave
"""

import os
import re
import string
import requests
import numpy as np
import collections
import random
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()

# Start a session
sess = tf.Session()


epochs = 300
batch_size = 250
max_sequence_length = 75
rnn_size = 10
embedding_size = 10
min_word_frequency = 1
learning_rate = 0.0005
num = 20

punctuation = string.punctuation
punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])

data = 'data_train.txt'
f = open(data, 'r')
x_data = f.read()
#x_data1 = x_data.replace('\r\n', '')
x_data1 = x_data.replace('\n', '')

def build_vocab(text, min_freq):
    word_counts = collections.Counter(text)
    # limit word counts to those more frequent than cutoff
    word_counts = {key: val for key, val in word_counts.items() if val > min_freq}
    # Create vocab --> index mapping
    words = word_counts.keys()
    vocab_to_ix_dict = {key: (i_x + 1) for i_x, key in enumerate(words)}
    # Add unknown key --> 0 index
    vocab_to_ix_dict['unknown'] = 0
    # Create index --> vocab mapping
    ix_to_vocab_dict = {val: key for key, val in vocab_to_ix_dict.items()}

    return ix_to_vocab_dict, vocab_to_ix_dict


# Build Shakespeare vocabulary
print('Building Shakespeare Vocab')
ix2vocab, vocab2ix = build_vocab(x_data1, min_word_frequency)
vocab_size = len(ix2vocab) + 1
print('Vocabulary Length = {}'.format(vocab_size))
# Sanity Check
assert (len(ix2vocab) == len(vocab2ix))


text_processed = []
for ix, x in enumerate(x_data1):
    try:
        text_processed.append(vocab2ix[x])
    except KeyError:
        text_processed.append(0)
text_processed = np.array(text_processed)
l = len(text_processed)
data_t = text_processed[0:int(l/2)]
data_label = text_processed[int(l/2):l]

train_data = np.zeros((int(num/2),max_sequence_length))
train_label = np.zeros((int(num/2),max_sequence_length))

j = 0
i = 0
for i in range(int(num/2)):
    k = 0
    
    while data_t[j] != vocab2ix['n']:
        train_data[i,k] = data_t[j]
        k = k+1
        j = j+1
        
    j = j + 1
    train_data[i,k] = vocab2ix['n']

    
j = 0
i = 0
for i in range(int(num/2)):
    k = 0
    
    while data_label[j] != vocab2ix['e']:
        train_label[i,k] = data_label[j]
        k = k+1
        j = j+1
        
    j = j + 1
    train_label[i,k] = vocab2ix['e']


def word_count(data):
    l = len(data)
    count = 0
    for i in range(l):
        if data[i] != 0:
            count = count + 1
    return count





def shift(data_in):
    i = 0
    while data_in[i]!=0 :
        i = i + 1       
    shape = np.shape(data_in)
    x = np.roll(data_in,max_sequence_length-i)
    #temp = np.zeros(shape)
    #temp = np.roll(x,-1)
    #temp[24] = m

    return x


def zero_print(data,ix2vocab,vocab2ix):
    seq = ''
    l = len(data)
    for i in range(l):
        x = data[i]
        if x!=0 :
            seq = seq + ix2vocab[x]
    print('\n',seq)
"""
temp = data_pro[0:1]
temp1 = np.reshape(temp,(25,1))
temp = shift(data_pro[0],4)
"""
"""
vocab_size = len(vocab2ix)
embedding_size = len(np.unique(data))
"""
identity_mat = tf.diag(tf.ones(shape=[embedding_size]))

x_data = tf.placeholder(tf.int64, [None, max_sequence_length])
y_output = tf.placeholder(tf.int64, [None])

x_embed = tf.nn.embedding_lookup(identity_mat, x_data)


if tf.__version__[0] >= '1':
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
else:
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size)

output, state = tf.nn.dynamic_rnn(cell, x_embed, dtype=tf.float32)
# Get output of RNN sequence
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([rnn_size, 5], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[5]))
logits_out = tf.matmul(last, weight) + bias
#out = tf.argmax(logits_out, 1)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output)
loss = tf.reduce_mean(losses)
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)
lo=[]
str_out = np.zeros((5,50))
for j in range(epochs):
 for i in range(10):
    x = train_data[i]
    x = shift(x)
    y = train_label[i]
    m = 1
    print('s \n')
    length = word_count(x)
    for k in range(length+5):
        x1 = np.reshape(x,[1,max_sequence_length])
        y1 = y[0:1]
        
        
        #y1 = np.squeeze(y1)
        m = y1
        train_dict = {x_data:x1,y_output:y1}
        sess.run(train_step,feed_dict=train_dict)
        out = sess.run(logits_out,feed_dict=train_dict)
        loss_1 = sess.run(loss,feed_dict=train_dict)
        out1 = np.argmax(out,axis=1)
        
        
        #lo.append(loss)
        x = np.roll(x,-1)
        x[max_sequence_length-1] = out1
        y = np.roll(y,-1)
       
    loss_2 = sess.run(loss,feed_dict=train_dict)
    lo.append(loss_2)
plt.plot(lo)       
plt.show()
data1 = 'data_test.txt'
f = open(data1, 'r')
x_data1 = f.read()
#x_data1 = x_data.replace('\r\n', '')
x_data1 = x_data1.replace('\n', '')
text_processed = []
for ix, x in enumerate(x_data1):
    try:
        text_processed.append(vocab2ix[x])
    except KeyError:
        text_processed.append(0)
text_processed = np.array(text_processed)
data_t = text_processed
l = len(text_processed)

test_data = np.zeros((6,max_sequence_length))



j = 0
i = 0
for i in range(int(6)):
    k = 0
    
    while data_t[j] != vocab2ix['n']:
        test_data[i,k] = data_t[j]
        k = k+1
        j = j+1
        
    j = j + 1
    test_data[i,k] = vocab2ix['n']
print('\n')  
print('test data sequences:')   
for i in range(6):
    seq = ''
    x = test_data[i]
    x = shift(x)
    y = test_data[i]
    temp = 1
    
    length = word_count(x)
    for k in range(length+7):
        x1 = np.reshape(x,[1,max_sequence_length])
        y1 = y[0:1]
        
        
        #y1 = np.squeeze(y1)
        #m = temp 
        train_dict1 = {x_data:x1,y_output:y1}
        #sess.run(train_step,feed_dict=train_dict)
        out = sess.run(logits_out,feed_dict=train_dict1)
        #loss_1 = sess.run(loss,feed_dict=train_dict)
        out1 = np.argmax(out,axis=1)
        temp = out1[0]
        m = temp
        """
        if temp!=0:
           seq = seq + ix2vocab[temp]
        else:
           seq = seq + ''
        """   
        #lo.append(loss)
        x = np.roll(x,-1)
        x[max_sequence_length-1] = out1
        y = np.roll(y,-1)
    
    zero_print(x,ix2vocab,vocab2ix)



    
    
    
