from lib4all import *
from ch6_1 import *

# ==== Hyper parameter 튜닝하기 ====

#holdout cross validation을 이용한 model 평가
#k-fold cross validation을 이용한 model 평가 (k-겹 교차 검증)

#overfitting & underfitting을 모두 피해야한다!
#위 valdiation들이 새로운 test set에 대해 얼마나 잘 작동하는지 밝혀준다!

"""
6.2.1 Holdout Cross Validation
"""

# 일반적으로 ML는 hyperparameter를 튜닝할 필요가 있다! => 이 과정을 모델 선택이라고 함(optimized hyperparameter를 선택하는 것)
# Test set은 고정되어있으므로, test set을 모델 선택에 반복하여 사용하면 overfitting됨
                #즉, Hyperparameter 튜닝에 test set을 사용하는 것은 좋은 선택이 아님
# 따라서 Data를 Train / Validation / Test set으로 나누어 validation set을 hyperparameter 튜닝에 사용!
# Train set의 일부를 validation set으로 나눔

# 단, 이 방법은 train-validation set을 어떻게 나누냐에 따라 성능이 민갑하게 변할 수 있다 (단점)
# 이를 보완한 것이 k-fold cross validation


"""
6.2.2 K-fold Cross Validation

"""

############### k-fold 개요 ##############
# 홀드아웃에서 트레이닝-벨리데이션 나누는거 편향을 줄이기 위해 도입 가능
# k-fold ==> Train Data Set을 k개로 분할
# k-1개로 모델 훈련, 나머지 1개로 성능 평가
# 이 과정을 통해 좋은 hyperparameter 값을 찾음

#그리고 이번에는 얻은 hyperparameter를 이용, Train data set 모두를 사용하여 재훈련 (dataset이 클수록 결과가 좋으므로)
#그 뒤 test set으로 검증!

# k-fold는 각각의 1/k train set이 한번씩 validation set으로 사용됨. 따라서 모델 성능 추정에 있어 분산이 낮음(결과가 크게 랜덤하게 나오지 않음)
# ==> 각 fold에 대한 성능을 평균내여 평균 성능으로 사용

############### k값에 따른 성능 ##############
# 경험적으로 10-fold가 보편적으로 권장할만함 (bias-variance trade-off상 권장)
# k가 크면 학습에 오랜 시간 걸림 + variance 증가(각 fold마다 서로 비슷해지기 때문)
# 큰 data set에서는 5-fold만 해도 충분히 괜찮은 성능 나옴

# LOOCV
# Special k-fold for small data set
# 작은 data-set에서는 Leave-One-Out Cross-Validation(LOOCV) 사용할 수도 있음
# 이는 k = n으로, train set의 각 sample을 모두 validation set으로 사용하는 것
# from sklearn.model_selection import LeaveOneOut

############### upgrade version for k-fold ##############
# stratified k-fold cross-validation
# : k-fold의 upgrade version. 이름에서 볼 수 있듯이, 각 fold의 class 비율을 train set의 class 비율과 동일하게 유지함

# Stratified k-fold cross-validation 구현

import numpy as np
from sklearn.model_selection import StratifiedKFold
print('\nk-fold 직접 구현\n')
kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train,y_train)
scores = []

for k, (train, test) in enumerate(kfold):
    #print("val =", test)  #test에는 test index가 저장된 list가 존재
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print(f'폴드: {k+1:2d},  '\
            f'클래스 분포: {np.bincount(y_train[train])},  '\
            f'정확도: {score:.3f}')

print(f'CV 정확도: {np.mean(score):.3f} +/- {np.std(scores):.3f}')

# k-fold와 같이 여러번 학습시켜야 하는 경우 pipeline이 굉장히 유용


# sklearn에 이미 구현되어있는 k-fold function 이용
print('\nk-fold scikit-learn 이용\n')
from sklearn.model_selection import cross_val_score

# scores = cross_val_score(estimator=pipe_lr,
#                         X=X_train,
#                         y=y_train,
#                         scoring=['accuracy'],
#                         cv=10,
#                         n_jobs=-1)#각 fold의 평가를 몇 개의 cpu에 분산할 것인가, -1 = all
# print(f'CV 정확도 점수: {scores}')
# print(f'CV 정확도: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
# 기본적으로 점수는 회귀는 R^2, 분류는 accuracy로 평가함.
# 바꾸고 싶으면 scoring을 바꾸면 됨(적지 않으면 default로 들어감)
from sklearn.model_selection import cross_validate
# scores = cross_validate(estimator=pipe_lr,
#                         X=X_train,
#                         y=y_train,
#                         scroing=['accuracy'],
#                         cv=10,
#                         n_jobs=-1,
#                         return_train_score=False)
# print(f'{ np.mean(scores['test_accuracy']) }')
#print()