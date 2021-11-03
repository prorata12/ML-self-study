from lib4all import *
from ch6_1 import *
from ch6_2 import *
from ch6_3 import *

"""
6.4.1 Grid Search - hyper parameter 튜닝
"""
# 두 종류의 parameter 존재
# 1. tuning parameter : 학습 과정에서 학습되는 parameter(LR의 가중치(w))
# 2. hyper parameter : 학습 과정에서 학습되지 않는 별도의 parameter

# Grid Search - list에 저장된 여러 hyper parameter 값들의 조합 전체를 조사함
# 그를 통해 최적의 조합을 찾아냄

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(),\
                        SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C':param_range, 'svc__kernel':['linear']},
                {'svc__C':param_range, 'svc__gamma': param_range, 'svc__kernel':['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=10,
                    n_jobs=-1)
gs = gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)

"""
6.4.2 Nested Cross Validation (중첩교차검증)
"""
# 만약 5-fold로 검증을 한다고 하면,
# 1 fold를 test fold로, 4-fold를 train fold로 나눈다.
# 이 train fold를 절반으로 나누어 두개의 fold를 다시 만든다(즉, 각각 2 fold 크기)
# 그 뒤 이 중 하나는 train fold로, 하나는 validation fold로 사용하여
# hyper parameter 튜닝을 한다
# 그 뒤 그 hyper parameter 값을 이용하여 밖에로 다시 나가서
# 4 fold를 train fold로 이용하여 학습하고 전에 나누었던 test fold를 통해 테스트한다.
# 이 경우, 처음에 원본 set을 5 fold로 나누고 그 뒤 train fold를 다시 2 fold로 나누었으므로 
# 5x2 cross validation이라고도 한다. 

# 즉, 바깥쪽 루프 각각마다 서로 다른 optimized hyperparameter에 대한 성능이 나온다.
# 이를 이용해 각각의 성능을 평균내면 처음 본 데이터에 대해 평균적으로 어느정도의 성능이 나오는 지 알 수 있다.
# 이 방식을 이용해서 서로 다른 두 알고리즘이 새로운 데이터 셋에 대해서 평균적으로 뭐가 더 좋은지 비교할 수 있다.