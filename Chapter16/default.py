# -*- coding: utf-8 -*-
# ignore tensorflow messeages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import models, layers

import inspect

# https://www.it-swarm-ko.tech/ko/python/%eb%b3%80%ec%88%98-%ec%9d%b4%eb%a6%84%ec%9d%84-%eb%ac%b8%ec%9e%90%ec%97%b4%eb%a1%9c-%ec%96%bb%ea%b8%b0/1040573605/

# def retrieve_name(var):
#     callers_local_vars = inspect.currentframe().f_back.f_locals.items()
#     return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def printing(arr):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    Vari = [var_name for var_name, var_val in callers_local_vars if var_val is arr]
    print(f'\n<Value of {Vari[0]}> - printing function')
    print(f'{arr}\n')

import pickle
def pickle_load(filename, var, write_mode = 0):
    """
    example filename
    './Chapter15/filename.pkl'
    """
    load_flag = 0
    #file이 존재하지 않는 경우 스킵 (load_flag = 0)
    if (not os.path.isfile(filename)):
        print('Variable does not exist. This is the first running.')
    #file이 존재하는 경우 불러옴 (load_flag = 1)
    else:
        #읽기 모드면
        print('File exist. Load variable from the file')
        with open(filename, 'rb') as pickle_file:
            var = pickle.load(pickle_file)
            load_flag = 1
    return load_flag, var

def pickle_save(filename,var):
    #file이 존재하지 않으면 그냥 저장
    if (not os.path.isfile(filename)):
        with open(filename, 'wb') as pickle_file:       #file을 쓰기모드로 생성하여
            pickle.dump(var, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) #덮어쓰기
    #존재하면 삭제 후 새로 저장
    else:
        os.remove(filename)
        with open(filename, 'wb') as pickle_file:       #file을 쓰기모드로 생성하여
            pickle.dump(var, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) #덮어쓰기