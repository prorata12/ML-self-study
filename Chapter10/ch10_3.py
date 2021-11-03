import pandas as pd

# 10.3.1 GD/SGD (Gradient Descent) 방법으로 Regression Model Parameter 구하기 (직접 구현)

# OLS : Ordinary Least Squares (최소제곱법)
# = Linear Least Squares (선형 최소 제곱법)

# OLS Regression : Adaline의 Cost fucntion과 동일(Sum of Squared Errors, SSE)
# 단, Adaline과 다르게 step function 없이 연속적인 결과를 얻음 (즉, unit step function만 제거한 것)

df = pd.read_csv('./Chapter10/housing.data.txt',header=None,sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', #범죄율, 큰주택비율, 비소매비율, 찰스강인접(0 or 1)
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', #일산화질소농도, 주택 방 수, 오래된 집비율, 고용센터까지거리, 고속도로까지접근성
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'] #제산세율, 학생-교사비율, 아프리카계미국인비율이 0.63에서 얼마나 먼지, 저소득비율, 주택의 중간가격

class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        #self.w_ = np.zeros(1 + X.shape[1]) # Num of Feature + 1 for w0
        rgen = np.random.RandomState()
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X): # get [transpose(w) * X]
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X): #주어진 X에 대해 그냥 예측값 return하는 함수
        return self.net_input(X)


# X,y scaling

X = df[['RM']].values
y = df['MEDV'].values

from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
# flatten : multi dimension array를 1차원 array로 변환
# newaxis -  [:,newaxis] : (4,) => (4,1) 
#            [newaxis,:] : (4,) => (1,4)

lr = LinearRegressionGD(eta=0.001)
lr.fit(X, y)

#Plot the cost of Model for each iteration(epoch)
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.title('SSE according to Epoch')
plt.show()

# Make helper function to plot prediction
# plot the helper function and scatterplot
def lin_regplot(X,y,model):
    plt.scatter(X,y,c='steelblue',edgecolor='white',s=70)
    plt.plot(X, model.predict(X),color='black',lw=2)
    return None
lin_regplot(X_std,y_std,lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.title('Prediction using lr')
plt.show()

#plt.scatter(X, y,color='black',lw=2)
#plt.show()

# See Prediction result from arbitrary X value

num_rooms_std = sc_x.transform([[5.0]])
price_std = lr.predict(num_rooms_std)
print(f'$1,000 단위 가격: {sc_y.inverse_transform(price_std)[0]:.3f}') # using scaler, inverse transform predicted label y'

# Scaled data의 linear regression 는 항상 y절편이 0이다! (X,y 모두 0으로 평균이 맞춰졌기 때문)
# 따라서 y절편을 업데이트 해줄 필요가 없다.

print(f'기울기: {lr.w_[1]}')
print(f'y절편: {lr.w_[0]}')




# 10.3.2 사이킷런으로 회귀 모델 가중치 추정

from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X,y)
print(f'기울기: {slr.coef_[0]:.3f}')
print(f'y절편: {slr.intercept_}')
lin_regplot(X,y,slr)
plt.title('Prediction using slr(import library)')
plt.show()
# 참고, X_std, y_std로 하면 거의 동일한 기울기/y절편이 나온다. 손으로 짠 것도 제대로 구현했다는 뜻

# 맨 마지막 Note 추후에 공부해보기
