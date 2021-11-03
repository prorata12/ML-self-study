#Decision Tree Learning
#Maximize Information Gain

#Three Metrics for impurity
# Gini impurity, Classification Error, Entropy

"""
3.6.1 Comparing Impurity metrics
"""
import matplotlib.pyplot as plt
import numpy as np

#binary class에서의 impurity
#which indicates impurity of a node

def gini(p):
    return p*(1-p) + (1-p)*(1 - (1-p))

def entropy(p):
    return -p * np.log2(p) - (1-p)*np.log2(1-p)

def error(p):
    return 1 - np.max([p,1-p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]

err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111) #1x1 space, 1st position

for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],\
                        ['Entropy', ' Entropy(scaled)', 'Gini Impurity', 'Misclassification Error'],\
                        ['-','-','--','-.'],\
                        ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x,i,label=lab, linestyle=ls, lw=2, color=c)

#ax.legend()
ax.legend(loc='upper center', bbox_to_anchor = (0.5, 1.15), ncol=2, fancybox=True, shadow=False)
#범례 표기
#참고링크 https://kongdols-room.tistory.com/87
#loc = 대략적인 위치, bbox = 정확한 위치(0~1, 0~1)의 범위가 그래프 내부
#ncol = 범례를 몇 개의 column으로 표시할지

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
#horizontal line
#axvline : vertical line
#x축과 평행인 선 그리기
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')


# fig2 = plt.figure()
# ax2 = plt.subplot(211)
# ax3 = plt.subplot(212)

#plt.show()

fig2 = plt.figure()

"""
3.6.2 Making Decision tree
"""
from ch3_6_dataset import *

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=10, 
                              random_state=1)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test)) # 행 방향으로 합침
y_combined = np.hstack((y_train, y_test)) # 열 방향으로 합침
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree, test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#tight_layout: graph 근처의 흰색 padding을 상당수 제거한다
plt.show()





fig3 = plt.figure()

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree,
                           filled=True, 
                           rounded=True,
                           class_names=['Setosa', 
                                        'Versicolor',
                                        'Virginica'],
                           feature_names=['petal length', 
                                          'petal width'],
                           out_file=None)
                           #out_file='Hi.dot') 
graph = graph_from_dot_data(dot_data)  # out_file = None --> dot file 생성 없이 바로 png 파일 생성
graph.write_png('tree.png')



"""
3.6.3 Random Forest
"""

# Two hyperparameters
# (1) number of bootstrap sample n 
# (2) number of randomly selected features d


# smaller n --> going to underfitting, bigger n --> going to overfitting
# recommended d ~= sqrt(num of features)


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='gini',
                                n_estimators=100,  #number of little tree
                                random_state=1,
                                max_samples=10,
                                max_features='auto', # default value = auto = sqrt(num of features)
                                n_jobs=-1) # use multicore of computer, -1 : use all cores
forest.fit(X_train, y_train)

plot_decision_regions(X_train, y_train, 
                      classifier=forest)#, test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()