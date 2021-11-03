
from lib4all import *

"""
7.1 Ensemble Learning(앙상블 학습)
"""

# ==== Ensemble Learning이란? ====

# 여러 분류기를 하나로 연결하는 것

# 1. 이진 클래스 분류(Binary Class)
# Majority voting(과반수 투표) -  # 이진 클래스 분류에서 사용되는 단어
# 여러 분류기(Classifier)가 예측한 Label중 과반수인 것 선택

# 2. 다중 클래스 분류(MultiClass)
# 다수결 투표(Plurality Voting) ==> 가장 많이 쓰이는  
# 여러 분류기(Classifier)가 예측한 Label중 가장 많이 투표된 것을 선택 (최빈값 : mode)

# 3. Ensemble Model 만들기
# Decision Tree, SVM, LR 등 여러 알고리즘 사용
# 즉, Decision Tree끼리 융합한 Random forest은 당연히 Ensemble Learning이다.

# 4. Class 분류하는 방법
# predicted class : y_hat =  mode {C1(x) ,C2(x), ... , Cm(x)} ; Ci(x) = i번째 Classifier
# +1, -1의 이진 분류의 경우 그냥 Sum(Ci(x))한뒤 >=0 이면 +1로, <0이면 -1로 구분하면 된다



# ==== 어째서 Ensemble 이 더 성능이 좋을까? ====

# 동일한 에러율 e의 n개의 분류기 가정 (분류기끼리는 서로 독립적)
# 에러율 : 잘못된 class로 분류할 확률
# 이라고 할 때, n개의 분류기 중 k개가 잘못 분류할 확률은
# e_ensemble = sigma(k>n/2) [e^k * nCk * (1-e)^(n-k)] (k개가 잘못 분류하고, n-k개가 잘 분류)
# 즉, 절반 이상이 잘못 분류할 확률이므로 몹시 낮아진다. (단, 다중 클래스에서는 계산식이 조금 더 복잡하기는 함)
# nCk : n choose k 로 읽음
# 단, 위 식을 직접 계산해보면 e < 0.5일때만 성능이 더 좋아진다! 즉, Random Guess보다는 성능이 좋아야 Ensemble이 좋다.
# 즉, 일반적으로 항상 성능이 더 좋아진다고 보면 된다.

