#K-mean clusterings
# : Prototype-based clustering이다.
# + Hierarchical clustering, density-based clustering
# Prototype : Centroid(평균)/medoid(최빈/대표값)을 이용함

# Clustering 평가 - elbow method / silhouette plot

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150, 
                  n_features=2, 
                  centers=3, 
                  cluster_std=0.5, 
                  shuffle=True, 
                  random_state=0)
                  
import matplotlib.pyplot as plt                  

plt.scatter(X[:, 0], X[:, 1], 
            c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.tight_layout()
plt.show()




#donwload this import sklearn_extra -> medoid

#11.1.4
# mu_i : 각 cluster에 속할 확률
# 처음에 각 cluster에 속할 확률은 랜덤하게 설정
# 각 sample마다 각 Cluster 소속 확률 업데이트
# Cluster 확률이 변하지 않거나/오차범위내/최대반복횟수내로반복

# 목적함수 = 모든, 각 sample i 에 대해서 모든, 각 cluster centroid j에 대한 거리 제곱합 * 각sample/cluster 마다 변하는 가중치 w(i,j)
# 즉, 실제 속하는 클러스터 w가 크고 실제 속하지 않는 w가 작아야 목적함수가 최소가 된다. [단, sigma w(i,:) = 1 , w(i,j) =i가 j cluster에 속할 확률, 즉 w를 통해 계산가능]
# w^m 을 씌워서, m이 커질수록 여러 cluster에 속할 확률이 커진다
# 















