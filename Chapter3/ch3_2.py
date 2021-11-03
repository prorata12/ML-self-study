from MLdefault import *
print = pset(0)

"""
Iris Dataset 불러오기
"""
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

print('클래스 레이블:', np.unique(y))

# for i,j in zip(iris.data,iris.target):
#     print(i,j)


"""
Dataset을 Train set - Test set으로 나누기
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X,y,test_size = 0.3, random_state = 1, stratify = y)

# 30%의 test set
# 데이터셋을 미리 섞음 --> random_state = 1에서 1 이 random seed로 작동,
# random_state를 고정함으로써 동일한 실행 결과를 재현할 수 있음
# stratify --> 계층화, test set / train set의 label 비율을 동일하게 맞춘다는 의미
# stratify에는 계층화할 class 정보,즉 여기서는 0,1,2의 label을 가진 y를 넣어준다
# y 내에 0,1,2가 각각 몇 개 있는지 확인하여 계층화해줌

print('y의 레이블 카운트: ',np.bincount(y)) #총 150개
print('y train의 레이블 카운트: ',np.bincount(y_train)) #총 105개
print('y test의 레이블 카운트: ',np.bincount(y_test)) #총 45개


"""
Dataset의 feature scaling (Normalization)
"""
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
print(sc.fit(X_train))
#StandardScaler.fit --> X_train의 mean, variation 계산
X_train_std = sc.transform(X_train)
#transform --> fit에서 계산한 mean, variation으로 Standardization
X_test_std = sc.transform(X_test)


"""
Perceptron 훈련
"""
from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter = 40, eta0 = 0.1, tol = 1e-3, random_state = 1)
#max iter = number of epoch
#eta = 학습률 -> 너무 크지도, 작지도 않게!
#tol = 허용 오차, 이 내로 들어가면 수렴했다고 생각하고 스탑 --> 너무 작으면 overfitting!
#나중에 재현할 수 있도록 random_state 사용 권장
#Scikit learn은 기본적으로 OvR(One versus Rest)방식으로 \
# multiclass classification 지원

ppn.fit(X_train_std, y_train)
#fit을 통해 linear model perceptron 훈련
#features(X) -> one perceptron(WX+b) -> prediction(y) [W 학습]
#Linear activation function, unit_step function 사용 ?

y_pred = ppn.predict(X_test_std)
#훈련된 model을 이용해 test set predict
print('잘못 분류된 샘플 개수: %d' % (y_test != y_pred).sum())


"""
성능 지표 (Performance Indicator) - 현 모델의 성능 측정
"""

from sklearn.metrics import accuracy_score
#Perceptron의 accuracy
print('정확도: %.2f' % accuracy_score(y_test, y_pred))

"""
Samples 시각화
"""