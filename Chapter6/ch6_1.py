from lib4all import *

import pandas as pd

# ==== pipeline을 만들어서 학습 간략화하기 ===

"""
6.1.1 데이터셋 불러와서 feature, label 저장한 뒤 train-test set 나누기
"""


"""
Step 0
Data Set 읽기
"""
# df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#                  'machine-learning-databases'
#                  '/breast-cancer-wisconsin/wdbc.data', header=None)

#Data Set 읽기
df = pd.read_csv('./Chapter6/wdbc.data', header = None)
print(df.head())
print(df.shape)


"""
Step 1
Data에서 feature-label 구분하기
"""

from sklearn.preprocessing import LabelEncoder

#feature, label 저장
X = df.loc[:,2:].values #feature
y = df.loc[:,1].values #label

#Label Encoding
le = LabelEncoder()
y = le.fit_transform(y)
print(le.classes_)

print(le.transform(['M','B'])) #M -> 악성, M -> 양성

"""
Step 2
Train-Test Set 구분하기
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X,y, test_size = 0.20, stratify = y, random_state = 1)

#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------


"""

6.1.2 pipeline을 이용한 Machine Learning Model을 Wraping하기 

"""

#많은 ML 알고리즘은 feature가 동일 scale을 가져야 좋은 성능이 나온다!
#여기서는 PCA를 통해 30dim --> 2dim으로 축소
#pipeline은 preprocessing 과정을 모두 pipeline 내로 묶어준다 (wraper)

from sklearn.preprocessing import StandardScaler        #Scaling
from sklearn.decomposition import PCA                   #Dimensionality Reduction
from sklearn.linear_model import LogisticRegression     #Classification
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(StandardScaler(),PCA(n_components=2), LogisticRegression(solver='liblinear', random_state = 1))

pipe_lr.fit(X_train, y_train) #StandardScaler로 scaling하고 mean, var 체크, PCA로 # of dim = 2 로 축소 후 LR로 분류해서 model 생성
                            #모두 fit_transform으로 작동
y_pred = pipe_lr.predict(X_test) # 학습된 model로 test set 분석하기, 여기서는 transform만 작동
print('테스트 정확도: %.3f' % pipe_lr.score(X_test,y_test)) #분석 결과 score로 체크하기



