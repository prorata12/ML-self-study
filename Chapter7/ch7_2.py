
from lib4all import *

"""
7.2 다수결 투표를 사용한 분류 앙상블
"""
# 실제로 구현해보자!

# 일반적으로 Plurality voting(다수결 투표, 멀티클래스)가 당연히 일반적(클래스>2인 경우가 많으므로)

"""
7.2.1
"""
# 각 분류 모델의 신뢰도에 가중치 부여하여 연결 가능
# 그냥 각 (모델 가중치 * Class_label)의 합이 가장 큰 Label로 할당 (binary의 경우) => np.argmax + np.bincount with weights로 가능
# Chi : 내부가 참이면 1, 거짓이면 0

# 확률이 잘 Calibration이 되어있다면 (즉, 해당 클래스일 확률이라고 나온 값(predict_proba return값)이, 정말로 해당 클래스일 확률인 경우, 즉 결과가 0.7일때 정말로 70%의 sample이 해당 label일때)
# 그냥 확률에 가중치 곱해서 더해도 가능(sum_j(w_j*p_i)) (j는 classifier, i는 class label)이 max가 되는 i를 찾으면 된다)
# argmax + np.average with weight로 가능
# 캘리브레이션 방법 :Platt's Scaling, Isotonic Regression

# Stacking != Plurality voting
# Stacking은 층이 2개인 Ensemble이라고 볼 수 있다.
# 첫번째 층의 예측 결과를 통해서 두번째 층(완전히 별개의 Classifier)를 학습시킨다.

# etc...
