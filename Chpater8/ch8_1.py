from lib4all import *

"""
Natural Language Processing(NLP) - Sentiment Analysis (= Opinion Mining)
자연어 처리 - 감정 분석
1. Text Preprocessing
2. Text --> Feature Vector
3. Text -> Positive or Negative Classification
4. Out-of-core learning (외부 메모리 학습) => 대용량 데이터 다루기
"""

#Use IMDb (Internet Movie DB)

# 8.1.1 - Download Text file

# 8.1.2 -  Preprocessing of Movie Review Data(Text Preprocessing)
# (1) Merge all each txt file into csv file
# (2) Read Movie Review and make it into DataFrame

import pyprind
import pandas as pd
import os
import numpy as np
first_run = 0
if first_run == 1:
# Only for the first time, merge all .txt file into .csv file
    basepath = r'C:\\Users\\user\\Desktop\\Projects\\ML\\Chpater8\\aclImdb'
    labels = {'pos': 1, 'neg':0}
    pbar = pyprind.ProgBar(50000)
    df = pd.DataFrame()

    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(basepath, s, l) # in aclImbda, there are pos/neg in each test/train
            for file in sorted(os.listdir(path)): # why sort? @@@
                with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                    txt = infile.read()
                df = df.append([ [txt, labels[l]] ], ignore_index = True ) # ignore Index? @@@
                pbar.update()
    df.columns = ['review', 'sentiment']

    # data is alinged with their label ==> permutation으로 섞기
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv('movie_data.csv', encoding = 'utf-8')

# After Merging

df = pd.read_csv('movie_data.csv',encoding='utf-8')
print(df.head(10))
print(df.shape)