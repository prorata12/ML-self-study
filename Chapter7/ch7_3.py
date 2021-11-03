
from lib4all import *

"""
7.3 배깅: 부트스트랩 샘플링을 통한 분류 앙상블
"""

# 부트스트랩 샘플링 == 중복을 허용한 랜덤 샘플링 , Random forest에서 사용한 방법
# 배깅 : bootstrap aggregating 라고도 함
# 서로 다른 부트스트랩 샘플을 이용해 각각의 Classifier를 이용하여 Plurality Voting을 함
# Random Forest는 배깅의 특수한 경우 (Feature의 일부만을 뽑아서 배깅하는 경우)
# 배깅은 정확도 향상 + Overfitting 정도 감소

# variance는 감소 가능, bias 감소는 어려움 ==> 편향이 낮은 pruning 없는 decision tree를 classifier로 이용하는 이유