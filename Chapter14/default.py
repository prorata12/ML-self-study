# ignore tensorflow messeages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



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
