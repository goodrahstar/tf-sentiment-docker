# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:51:16 2017

@author: Rahul.kumar
"""

#! /usr/bin/env python


import data_helpers2 as data_helpers
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from datetime import datetime

# Parameters
# ==================================================

# Eval Parameters
batch_size= 64
path ='/root/w2v_cnn_sentiment/'
checkpoint_dir = "./runs/1487512553/checkpoints"
vocab_file =  "./vocab1487512553.json"

# Misc Parameters
allow_soft_placement = True
log_device_placement =  False
#query = "How many Mass Addition lines are currently present?"


# Load data. Load your own data here
print("Loading data...")
#x_test, y_test, vocabulary, vocabulary_inv = data_helpers.load_data(
#    eval=True, vocab_file=vocab_file,
#    cat1="./data/sentiment.positive", cat2="./data/sentiment.negative")

def dumpData(file_path, file_name, column, new_data):
#    new_data = [(searchme),(urls),(titles),api]
    datasetExist = False
    if os.path.exists(os.path.join(file_path, file_name)):
        datasetExist = True
        
    if not datasetExist:
        print 'Dump file not found: Creating dump file'
        pd.DataFrame(columns = column).to_csv(os.path.join(file_path, file_name) , index =False)
        
    data = pd.read_csv(file_path + file_name)
    new_df = pd.DataFrame([new_data], columns=list(data.columns))

    data = data.append(new_df,ignore_index=True)
    data.to_csv(file_path + file_name, index =False)
    print '---->Dumping Succesful'
    
x_test, y_test, vocabulary, vocabulary_inv = data_helpers.load_data(
eval=True, vocab_file=vocab_file,cat1= "./data/sentiment.positive", cat2= "./data/sentiment.negative")

# Evaluation
# ==================================================
#checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint_file = "./runs/1487512553/checkpoints/model-18200"
#print("checkpoint file: {}".format(checkpoint_file))
#graph = tf.Graph()
#with graph.as_default():
session_conf = tf.ConfigProto(
    allow_soft_placement=allow_soft_placement,
    log_device_placement=log_device_placement)
sess = tf.Session(config=session_conf)
#    with sess.as_default():

# Load the saved meta graph and restore variables
saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
saver.restore(sess, checkpoint_file)

# Get the placeholders from the graph by name
input_x = tf.get_default_graph().get_operation_by_name("input_x").outputs[0]
dropout_keep_prob = tf.get_default_graph().get_operation_by_name("dropout_keep_prob").outputs[0]

# Tensors we want to evaluate
predictions = tf.get_default_graph().get_operation_by_name("output/predictions").outputs[0]
                                                  

def engine(query = " " , senderid = '', senderName = '' , x_test=x_test):#, x_test,y_test,vocabulary,vocabulary_inv):    

    query = query
    new_question = query.strip()
    new_question = data_helpers.clean_str(new_question)
    new_question = new_question.split(" ")
    
    num_padd = x_test.shape[1] - len(new_question)
    new_question = new_question + ["<PAD/>"] * num_padd
    #print new_question
    for word in new_question:
        if not vocabulary.has_key(word):
            new_question[new_question.index(word)] = "<PAD/>"
        if 'product' in word or 'products' in word:
            new_question[new_question.index(word)] = "<PAD/>"
            
    #print new_question
                  
    x = np.array([vocabulary[word] for word in new_question])
    x_test = np.array([x])
        
    
#    y_test = np.argmax(y_test, axis=1)
    #print("Vocabulary size: {:d}".format(len(vocabulary)))
    #print("Test set size {:d}".format(len(y_test)))
    
    #print("\nEvaluating...\n")
    
    
    # Generate batches for one epoch
    batches = data_helpers.batch_iter(x_test, batch_size, 1, shuffle=False)

    # Collect the predictions here
    all_predictions = []

    for x_test_batch in batches:
        batch_predictions = sess.run(
            predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
        all_predictions = np.concatenate(
            [all_predictions, batch_predictions])
    
    # Print accuracy

    #print 'prediction---', int(all_predictions)
    if int(all_predictions[0]) == 1:
        print 'Positive'
        return      {'Name':  senderName,'Sentiment' : 'Positive' , 'Response' : 'Thank you for your valuable feedback! \n \nIt will help us to serve better in future.\n'}
    elif int(all_predictions[0]) == 0:
        print 'Negative'
        return     {'Name':  senderName,'Sentiment' : 'Negative' , 'Response' : 'We are really sorry for the bad experience with our product.\n   '}
    else :
        print 'Unidentified'
        return     {'Name':  senderName,'Sentiment' : 'Unidentified'}

#print engine(query = 'i love it' , senderName ='Rahul')
