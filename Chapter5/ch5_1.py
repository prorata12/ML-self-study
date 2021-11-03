import pandas as pd
df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data',header=None)

from sklearn.model_selection import train_test_split

#train-test set 나누기
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:,0].values
# X:features, y : index

X_train, X_test, y_train, y_test = \
    train_test_split(X,y, test_size=0.3,
    stratify=y,
    random_state=0)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

import numpy as np

cov_mat = np.cov(X_train_std.T)
#np.cov는 각 행에 한 feature값만 들어있을 것을 기대하므로 transpose
eigen_values, eigen_vecs = np.linalg.eig(cov_mat)


### 5.1.3
import matplotlib.pyplot as plt
tot = sum(eigen_values)
var_exp = [ i/tot for i in sorted(eigen_values, reverse=True)]
#reverse - sorted에서 제공하는 descending order 여부 결정
#explained variance

cum_var_exp = np.cumsum(var_exp) # return cumulative explained variance

plt.bar(range(1,14), var_exp, alpha=0.2,
    align='center', label = 'individual explained variance')
plt.step(range(1,14), cum_var_exp,
    where='mid', label = 'cumulative explained variance')

plt.legend(loc='best')
plt.show()

### 5.1.4

#투영행렬(Project Matirx - 차원 줄여주는 행렬) W 만들기
#eigen vectors는 한 열이 하나의 eigen vector!

#tuple of (eigenvalue, eigenvector)
eigen_pairs = [(np.abs(eigen_values[i]), eigen_vecs[:,i])
    for i in range(len(eigen_values))]

#eigenvalues를 기준으로 descending order sort
eigen_pairs.sort(key= lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:,np.newaxis],
                eigen_pairs[1][1][:,np.newaxis]))
# w : 각 열이 eigenvector인 matrix!
# 어떤 값 X(n-dimensional)을 eigenvector(size = 1)와 곱했을 때,
# 되는 값은 eigenvector에 X를 projection(정사영)했을 때의 길이이다.
# 즉, w[:,0]X = X의 eigenvector 방향(axis) 값이다
# 따라서 wX는 각 eigenvector를 basis로 한 차원에서
# X를 각 eigenvector방향으로 정사영 내렸을 때의 값들이므로
# 새로운 차원에서의 (x1,x2,x3,....,xn) 값이 된다(즉 새 차원에서의 좌표값이 된다)

a = X_train_std[0].dot(w) #실제로는 행렬 생김새상 XW를 해주는 것이 맞다
print(a)
X_train_pca = X_train_std.dot(w)

colors = ['r','b','g']
markers = ['s','x','o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0],
                X_train_pca[y_train==l, 1],
                c=c, label=l, marker = m)
    # y_train==3와 같이 하면
    # y_train 과 크기는 동일한 list가 리턴되는데
    # element 중 값이 3인 element는 True,
    # 3이 아닌 element는 False값이 들어가 있게된다.
    
    # X_train_pca[y_train==3, 0] 와 같이 하면,
    # List가 리턴되는데 y_train==3이라는 list에서
    # 값이 True인 경우 X_train_pca의 해당 index의 element 값이
    # List로 들어간다.
    # 잘 모르겠는데 해보면 알듯
plt.show()


### 5.1.5
# 이제 편하게 구현된 PCA를 사용하자!

#from matplotlib.color import ListedColormap

from lib4all import *

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr = LogisticRegression(solver='liblinear',multi_class='auto')
lr.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier = lr)
plt.show()

plot_decision_regions(X_test_pca, y_test, classifier = lr)
plt.show()

print(pca.explained_variance_ratio_)
pca = PCA(n_components=None) #None으로 하면 모든 eigenvector를 사용한다, 즉 차원 축소 진행 안됨
X_train_pca = pca.fit_transform(X_train_std)
print(pca.explained_variance_ratio_)
