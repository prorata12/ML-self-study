from MLdefault import *
from ch3_2 import *
print = pset(1)

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt



def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):
    markers = ('s','x','o','^','v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #==============결정 경계 그리기(decision boundary?)================

    x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() + 1
    #feature 1,2 의 min, max 찾기

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # (xx1, xx2)는 사각형 x1_min~x1_max, x2_min ~x1_max 사이의 모든 격자들
    Z = classifier.predict( np.array([xx1.ravel(), xx2.ravel()]).T )
    print(Z)
    # ravel : 다차원 행렬을 1차원으로 만들어주는 함수(행끼리 쭉 이어붙임)
    # 따라서 # np.array([xx1.ravel(), xx2.ravel()]).T 는 (xx1, xx2) 모든 페어 \
    # 사각형 내 격자들의 (x,y) 좌표들을 모두 모은 것
    # 그냥 meshgrid~ravel, T까지의 과정을 하나의 묶음으로 보면 될듯

    #그렇게 Z는 주어진 sample 들의 feature_min ~ feature_max 사이 모든 격자들에 대해
    #각 격자들이 어떤 결과를 갖는지 예측한 것
    
    Z = Z.reshape(xx1.shape)
    #이제 예측을 했으면, 다시 (x,y)형태, 즉 각 격자들이 어떤 예측값을 갖는지
    #행렬 형태로 변환함

    
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap=cmap)
    #xx1, xx2 좌표마다 Z의 값을 가질 때의 등고선을 그림
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    #feature1, feature2의 범위가 잘 드러나도록 그래프에서 표현할 x,y값을 잘라줌

    for idx, cl in enumerate(np.unique(y)):
        #cl : class 0,1,2,...
        #각 클래스마다 순서대로 scatter함
        #X[a,b]는 a번째 sample의 b번째 feature 여야하지않나?????
        plt.scatter(\
            x = X[y == cl, 0], y = X[y == cl,1],\
            alpha = 0.8, c = colors[idx],\
            marker = markers[idx], label = cl,
            edgecolor = 'black')

    # 테스트 샘플 그리기

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:,0], X_test[:,1],\
            c='',edgecolor='black',alpha=1.0,\
                linewidth=1, marker='o',
                s=100, label='test set')

if __name__ == '__main__':   
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train,y_test))
    plot_decision_regions(X=X_combined_std,y = y_combined,classifier=ppn,test_idx=range(105,150))
    plt.show()