
#mmodel 평가 - precision, accuracy, recall, F1

"""
6.5.1
"""

# confusion matrix
# TP, TF --> accuracy
# FP, FN 도 존재
#             예측 클래스
#                 N   P
# 실클래스     N  TN  FP
#             P  FN  TP
                
#                 0   1
#             0
#             1

"""
6.5.2
"""

# 오차 (ERR)   - FP+FN / ALL (얼마나 잘못 분류했는지)
# 정확도 (ACC) - TP+TN / ALL (얼마나 잘 분류했는지)
# ERR + ACC = 1

# 진짜양성비율 TPR - TP / P (진짜양성중 양성으로 분류된 비율)
# 거짓양성비율 FPR - FP / N (진짜음성중 양성으로 분류된 비율)
# FP를 낮춰야 환자의 걱정 줄일 수 있음
# TP를 늘려야 목숨이 위험할 확률을 낮출 수 있음

# 정확도 (PRE) - TP / TP + FP 양성으로 분류된 것 중 진짜 양성의 비율
# 재현율 (REC) - TP / P (진짜 양성중 양성으로 분류된 비율 == TPR)

# F1-점수 - 실전에서 많이 사용, 2(PRE*REC)/(PRE+REC) ==> 조화평균

# make-scorer로 자신만의 score 함수 정의 가능

"""
6.5.3
"""
# ROC(Receiver Operating Characteristic) Graph
# classificer의 확률 임계값, 즉
# 최종 결과에서 0~1사이 중 몇 이상이면 true고 아니면 false인 그 임계값을 바꾸면서
# ROC를 그림