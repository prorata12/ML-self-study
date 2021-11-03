# -*- coding: utf-8 -*-
from default import *

"""
16.4 다층 RNN으로 IMDb  영화 리뷰의 감성 분석 수행
"""

"""
16.4.1 데이터 준비
데이터 불러와서 / 각 단어의 등장횟수 카운팅 / 각 단어를 int로 mapping(re-label)
sequence 길이를 동일하게 만들기(w/ padding)
"""

import pyprind
import pandas as pd
from string import punctuation
import re
import numpy as np

df = pd.read_csv('./Chapter16/movie_data.csv', encoding='utf-8')
print(df.head(3))
#review : 리뷰 내용
#sentiment : 해당 리뷰의 감정(0,1 label)

from collections import Counter

counts = Counter()

"""
review file 불러오기 / 단어 등장횟수 count
"""
# --- pickle load --- #
load_flag, counts = pickle_load('./Chapter16/counts.pkl', counts)
load_flag2, df = pickle_load('./Chapter16/df.pkl', df)
assert load_flag == load_flag2
# --- pickle load end --- #


if load_flag == 0: #pickle. 파일이 존재하지 않으면 실행

    pbar = pyprind.ProgBar(len(df['review']),
                        title='단어의 등장 횟수를 카운트합니다.')
    for i,review in enumerate(df['review']):
        text = ''.join([c if c not in punctuation else ' '+c+' ' \
                        for c in review]).lower() #이어져있는 단어 별, punctuation 별 나뉘어있는 text 하나 생성
                                                    # e.g. text = "Hello , I ' m apple . "
        df.loc[i,'review'] = text # i라는 이름의 행, 'review'라는 이름의 열에 해당하는 cell에 추가 [df의 경우 i가 행 index와 동일]
        pbar.update()
        counts.update(text.split()) #text에 있는 내용을 counts에 추가하여 셈 (count 횟수가 더해짐)
                                    #counts.update('string') <--> counts.subtract('string') / Counter() 관련 method
                                    #counts.most_common(int) : int 개수만큼 가장 많은 word 출력
                                
    # --- pickle save --- #
    pickle_save('./Chapter16/counts.pkl', counts) #pickle
    pickle_save('./Chapter16/df.pkl', df)
    # --- pickle save end --- #


#print(counts) # {단어, 등장횟수} 의 dictionary

"""
create mapped_review
mapped_reviews : {정수index word : 등장횟수} 의 리스트
"""
## 고유한 각 단어를 정수로 매핑하는
## 딕셔너리를 만듭니다.
word_counts = sorted(counts, key=counts.get, reverse=True) #전체 review에서 등장횟수 많은 순으로 word를 sort
print(word_counts[:5])
word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)} #enumerate의 두번째 숫자는 시작 숫자. ii는 1부터 시작함

mapped_reviews = []
pbar = pyprind.ProgBar(len(df['review']),
                       title='리뷰를 정수로 매핑합니다.')

loaded, mapped_reviews = pickle_load('./Chapter16/mapped_reviews.pkl', mapped_reviews) #pickle
if loaded == 0: #pickle
    for review in df['review']: # review에는 punctuation 근처에 띄어쓰기가 추가된 text가 존재
        mapped_reviews.append([word_to_int[word] for word in review.split()]) # word -> word_index의 dictionary를 이용하여 review를 integer index의 집합으로 만듦
                                                                              # e.g mapped_review[0] = [24, 12, 3, 4, 12, 45]
                                                                              # 해당 index의 단어 순으로 등장, 숫자가 작을수록 Counter가 높은 단어
        pbar.update()
    pickle_save('./Chapter16/mapped_reviews.pkl', mapped_reviews) #pickle

#mapped_reviews : [{정수index word : 등장횟수} 의 리스트  (한 review의)] 의 리스트(review들 전체)

"""
각 review를 같은 길이의 sequence로 만듦
: 0 padding (짧은 경우)
: last 200 words (긴 경우)
"""


## 동일 길이의 시퀀스를 만듭니다.
## 시퀀스 길이가 200보다 작으면 왼쪽에 0이 패딩됩니다.
## 시퀀스 길이가 200보다 크면 마지막 200개 원소만 사용합니다.

sequence_length = 200  ## (RNN 공식에 있는 T 값 입니다)
sequences = np.zeros((len(mapped_reviews), sequence_length), dtype=int) # sequen_length의 0으로 패딩된 array를, review 개수만큼 생성
loaded, sequences = pickle_load('./Chapter16/sequences.pkl', sequences)
if loaded == 0:
    for i, row in enumerate(mapped_reviews): # i는 몇번째 리뷰인지, row는 review text(정수화된)
        review_arr = np.array(row) #python array를 np.array로 바꿈
        sequences[i, -len(row):] = review_arr[-sequence_length:] # sequences에 각 review를 저장. 오른쪽에서 -sequence_length개만큼. -로 초과하는 것은 영향을 끼치지 않음.
                                                                # 따라서 받아올/덮어쓸 크기만큼만 적어주면 됨.
    pickle_save('./Chapter16/sequences.pkl', sequences)

X_train = sequences[:37500, :] #numpy의 index는 맨 끝 index를 미포함(파이썬과 동일)
y_train = df.loc[:37499, 'sentiment'].values #data frame의 index는 처음/끝 index를 모두 포함. / 해당 label을 모두 포함한다고 보면 됨
X_test = sequences[37500:, :]
y_test = df.loc[37500:, 'sentiment'].values

#X_temp = sequences.loc[:37499, 'review'].values
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(type(sequences), type(df))

n_words = len(word_to_int) + 1 # padding이 0이므로, [0] + word_to_int 가 총 words 개수
print(n_words)

"""
16.4.2 임베딩 - 원핫인코딩 대신, 실수벡터로 변환 (저차원으로 바꿈)
=> 장점 (1) 차원 축소 : 차원의 저주 영향 감소
        (2) 워드->벡터 를 해주는 임베딩 층도 함께 학습, 중요한 feature 추출이 가능
"""
print('\n\n ===== 16.4.1 =====\n\n')

from tensorflow.keras import models, layers
model = models.Sequential()
model.add(layers.Embedding(n_words, 200, 
                           embeddings_regularizer='l2')) #200차원 vector로 단어를 embedding함, embedding_initializer로 가중치 다른 방식으로 초기화 가능(기본은 uniform)
print(model.summary()) #(None, None, 200) : batch size, time steps, dimensionality
# 파라미터 수 = 102967 * 200 (이전 차원 크기 * 현재 차원 크기)

"""
16.4.3 RNN 모델 만들기
"""
model.add(layers.LSTM(16)) # 16개의 순환유닛 존재(16개 node)
                           # activation -> h(hidden state)에서 사용할 활성화 함수(tanh)
                           # recurrent_activation -> C(cell state)에서 사용할 활성화 함수(hard_sigmoid( x > |2.5| 일때는 0, 그 사이에서는 0.2x + 0.5))
                           # hard sigmoid는 계산 효율성때문에 사용, 이후 버전에서는 기본값이 sigmoid가 될 수도 있다(최근 지원중이기 때문)

                           # dropout -> hidden state를 위한 드롭아웃 비율(default = 0)
                           # recurrent_dropout -> cell state를 위한 dropout 비율
                           # return_sequences = True (두 개 이상의 순환층을 쌓는 경우, 아래층의 모든 time step이 위층으로 전달되어야 함. 이때 True로 설정)
                           # -> 기본적으로 RNN은 마지막 출력이 output이기 때문에 마지막 time step의 hidden state만을 출력함. 위의 True세팅을 통해 모든 time step state를 위로 넘겨줄 수 있음
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
model.add(layers.Flatten())
#Recurrent layer를 Dense layer에 연결하기 위해서는 Recurrent Layer를 이를 Flatten해줘야한다. [CNN에서와 동일]
model.add(layers.Dense(1, activation='sigmoid')) #최종 output이 1차원 값이므로 unit은 하나.

print(model.summary())
#13888 = 200*16 (x->h) + 16*16 (h(t-1) -> h(t)) + 16 (bias for h) = 3472
#      = 3472 * 4 (삭제게이트(f) 인풋게이트(i,g), 아웃풋게이트(o))

"""
16.4.4 모델 훈련하기
"""


model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['acc'])

import time
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
if (not os.path.isfile('./Chapter16/sentiment_rnn_checkpoint.h5')):
    callback_list = [FCH(filepath='./Chapter16/sentiment_rnn_checkpoint.h5',
                                    monitor='val_loss', 
                                    save_best_only=True)]

    history = model.fit(X_train, y_train, 
                        batch_size=64, epochs=10, 
                        validation_split=0.3, callbacks=callback_list)
    pickle_save('./Chapter16/history_history.pkl',history.history)
else:
    model.load_weights('./Chapter16/sentiment_rnn_checkpoint.h5')
    history_history = 0
    history_history = pickle_load('./Chapter16/history_history.pkl', history_history)
    
import matplotlib.pyplot as plt
print(type(history_history))
print(history_history)
epochs = np.arange(1, 11)
plt.plot(epochs, history_history[1]['loss'])
plt.plot(epochs, history_history[1]['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

epochs = np.arange(1, 11)
plt.plot(epochs, history_history[1]['acc'])
plt.plot(epochs, history_history[1]['val_acc'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

"""
16.4.5 감성 분석 RNN 모델 평가
"""
model.evaluate(X_test, y_test) # 기본적으로 loss 출력.
#model.compile(loss='binary_crossentropy', 
#              optimizer='adam', metrics=['acc']) 여기서 metrics에 다른거 추가하면 추가 반환

model.predict_proba(X_test[:10]) #양성 클래스(1), 즉 긍정 리뷰일 확률 반환, predict method와 동일
model.predict_classes(X_test[:10]) # 예측 클래스를 리턴함
# LSTM의 유닛 개수, 타임 스텝 길이, 임베딩 크기 조절을 통해 더 좋은 성능 얻을 수 있음.