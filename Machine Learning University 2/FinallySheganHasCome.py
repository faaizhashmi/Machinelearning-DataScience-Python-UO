# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:31:31 2019

@author: faaiz
"""
import numpy as np
import cvs

def read_data(filename):
    raw_data=open(filename,'rt')
    csv.reader(raw_data,delimeter=',')
    x=list(data)
    return np.array(x).astype('float')

data=read_data('iris-coded.csv')