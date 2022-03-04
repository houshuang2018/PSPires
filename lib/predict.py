#!/usr/bin/env python
# Date: 2022-02-25
# Author: Shuang Hou
# Contact: houshuang@tongji.edu.cn

import os
import sys
import numpy as np
import pandas as pd
import pickle
import bz2
import random
from lib.Utility import *

from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
    
# ------------------------------------
# functions
# ------------------------------------
def preprocess_data(features,instance):
    scaler = QuantileTransformer()
    data = features.copy()
    data = data.fillna(0)
    y = data[instance]
    data = data.drop([instance], axis=1)
    data = data._get_numeric_data()
    columns = data.columns
    data = scaler.fit_transform(data)    
    data = pd.DataFrame(data, columns=columns)
    data[instance] = y
    return data, scaler


def get_test_train_indexes(data, label, ratio=1, seed=42):
    '''
    Function: Oersample the positive data with randomly selected negative samples.
    Returns: List with indexes which contain positive and negative samples.
    Reference: van Mierlo et al., 2021
    '''
    random.seed(seed)
    positive_instances = set(data.loc[data[label] == 1].index)
    negative_instances = set(data.loc[data[label] == 0].index)
    n_positives = data.loc[data[label] == 1].shape[0]    
    indexes = list()    
    while len(negative_instances) >= 1:    
        if len(negative_instances) > n_positives * ratio:
            sample_set = random.sample(negative_instances, (n_positives * ratio))
        else:
            sample_set = list(negative_instances)
            size = (len(sample_set))
            short = (len(positive_instances ) * ratio) - size        
            shortage = random.sample(set(data.loc[data[label] == 0].index), short)
            sample_set = (sample_set+ shortage)
        indexes.append((list(positive_instances) + list(sample_set)))
        negative_instances.difference_update(set(sample_set))
    return(indexes)


def train_model(instance, random, modelFile, trainData, jobs):
    df = trainData.select_dtypes([np.number])
    indexes = get_test_train_indexes(df, instance)
    modlist = list()
    clf = RandomForestClassifier(n_jobs=jobs, class_weight="balanced", n_estimators=1200, criterion="entropy", random_state=random)
    for index in indexes:
        df_fraction = df.loc[index]        
        X = df_fraction.drop(instance, axis=1)        
        y = df_fraction[instance]
        clf.fit(X, y)
        modlist.append(clf)
    pickle.dump(modlist, open(modelFile, 'wb'))
    

def predict(outputFile, tmpDir, oridata, random, modelFile, trainDataFile, jobs):
    pd.set_option('mode.chained_assignment', None)
    
    if len(oridata) == 0:
        raise_error('The features is empty. Can not predict.')
        
    ## get training data
    if not os.path.isfile(trainDataFile):
        raise_error(f'{trainDataFile} does not exists. Please check the file.')
    else:
        trainData = bz2.BZ2File(trainDataFile, 'rb')
        trainData = pickle.load(trainData)
        
    overlap = list(set(trainData['uniprot_id'].tolist()) & set(oridata['uniprot_id'].tolist()))
    overlap = ','.join(overlap)
    info(f'The following ids are in training data: {overlap}')
    
    instance = 'llps'
    trainData, scaler = preprocess_data(trainData, instance)
    
    uid = oridata.select_dtypes(include='object')
    data = oridata.copy()
    data = data._get_numeric_data()
    columns = data.columns
    data = scaler.transform(data)
    data = pd.DataFrame(data, columns=columns)
    exportData = uid.merge(data, how='outer', left_index=True, right_index=True)
    exportData.to_csv(tmpDir + '/transformed_features.csv', index=False)
        
    ## train model
    if random != 42:
        modelFile = tmpDir + '/models_seed' + str(random) + '.sav'
        train_model(instance, random, modelFile, trainData, jobs)
    
    ## predict
    prediction = {}
    if not os.path.isfile(modelFile):
        raise_error(f'{modelFile} does not exists. Please check the file.')
    else:       
        modlist = pickle.load(open(modelFile, 'rb'))
        num = 0
        for model in modlist:
            probability = model.predict_proba(data)[:, 1]
            prediction['probability_'+str(num)] = probability
            num += 1
    prediction = pd.DataFrame(prediction)
    scores = uid.merge(prediction, how='outer', left_index=True, right_index=True)
    scores.to_csv(tmpDir + '/prediction_scores.csv')
    
    ## calculate mean value to get protein phase separation probability score
    tmp = pd.DataFrame(np.mean(prediction, axis=1))
    outData = uid.merge(tmp, how='outer', left_index=True, right_index=True)
    
    otable = outData.to_csv(index=False)
    outputFile.write(otable)
