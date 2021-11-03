from ch3_6_dataset import *


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, 
                           p=2, 
                           metric='minkowski')

# neighbors 수가 적으면 overfitting! 많으면 underfitting!
# lp norm. p=2이면 l2 norm(Euclidian distance 사용)
# minkowski : use norm for distance
# About Metric
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

knn.fit(X_train_std, y_train)

# need standardized data! // y는 label이므로 standardize 하지 않음 유의
# knn의 경우 standardization이 거의 필수임에 유의할것(각각의 거리를 따지므로)
# knn은 high-dimensional features에 대해서는 잘 작동하지 않는다!
# 이 경우 features selection / dimensionality reduction 사용
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
#

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=knn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()