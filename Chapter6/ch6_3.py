from lib4all import *
from ch6_1 import *
from ch6_2 import *

"""
6.3.1 학습 곡선을 이용한 over/underfitting 여부 판단 - 모델 자체 평가
"""
# train sample 개수에 따른 학습 accuracy를 통해 모델이 괜찮은지 판단
# Overfitting 되었는지, underfitting 되었는지 판단하기


# 1. train sample 수가 늘어나도 train/test accuracy가 모두 낮은 값에 수렴하는 경우
# --> underfitting
# 해법: parameter 수 늘리기(feature 추가수집, 규제 감소)
# 2. train sample 수가 늘어나도 train/test accuracy가 잘 수렴하지 않을 때
# --> overfitting
# 해법: parameter 수 줄이기(feature selection/extraction), 규제 증가
# 해법: or sample 수 늘리기(단, noise가 많거나 거의 최적화된경우에는 소용없음)
#           --> 이 경우에 대한 문제는 다음 절 추가 논의


"""
6.3.2 검증 곡선을 이용한 over/underfitting 여부 판단 - hyper parameter 튜닝
"""
#hyper parameter에 따른 모델 성능 측정