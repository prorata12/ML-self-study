"""
10.2 주택 데이터셋 탐색
"""
def ismain():
    return __name__ == '__main__'
# 10.1.1 주택 데이터셋 읽기
print('\n =====10.2.1===== \n')
import pandas as pd
df = pd.read_csv('./Chapter10/housing.data.txt',header=None,sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', #범죄율, 큰주택비율, 비소매비율, 찰스강인접(0 or 1)
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', #일산화질소농도, 주택 방 수, 오래된 집비율, 고용센터까지거리, 고속도로까지접근성
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'] #제산세율, 학생-교사비율, 아프리카계미국인비율이 0.63에서 얼마나 먼지, 저소득비율, 주택의 중간가격
if ismain():
    print(df.head())

# 10.2.2 데이터셋 특징 시각화
# Linear Regression Traning에 있어서 Data의 Feature/Target이 정규분포일 필요는 없다!

# 탐색적 데이터 분석(Exploratory Data Analysis(EDA)) : 모델 훈련 전 가장 먼저 할 것을 권장. 데이터 분포 시각화 및 분석 작업
#산점도 행렬(scatterplot matrix) : 데이터셋 내 특성간의 상관관계를 나타냄

print('\n =====10.2.2===== \n')
import matplotlib.pyplot as plt
import seaborn as sns


cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

sns.pairplot(df[cols], height=2)
plt.tight_layout()
if ismain():
    print('plot the scatterplot matrix')
    plt.show()

# 10.2.3 상관관계 행렬을 이용한 분석
# 상관관계 행렬 : Correlation Matrix
# Correlation Matirx = Scaled Covariance Matrix (공분산 행렬을 scale조절 = 상관관계 행렬)
# 실제로 계산해보면 ...
# Correlation coefficient of raw data == Covariance of scaled data(스케일조정된 data)
# cov(x',y') = cov(x,y)/[var(x)*var(y)]
# x', y' = scaled x,

# Pearson product-moment correlation coefficient (피어슨의 상관관계 계수 / 피어슨의 r)
# -1 <= r <= 1, 1: 양의 상관관계, 0: 상관관계 없음, -1: 음의 상관관계

print('\n =====10.2.3===== \n')

import numpy as np


cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
if ismain():
    print('plot correlation coefficient using heatmap')
    plt.show()

# Target Variable : MEDV 를 기준으로 상관관계 높은 특성을 찾자
# 확인하면 LSTATS ~ MEDV 는 음의 상관관계 확실하지만(10.2.3) 비선형적(10.2.2)
# RM ~ MEDV 는 양의 상관관계 확실하며 선형적
# 따라서 RM이 간단한 Linear Regression에 있어 좋은 특성으로 보임



