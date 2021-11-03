from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([258.0, 270.0, 294.0, 
              320.0, 342.0, 368.0, 
              396.0, 446.0, 480.0, 586.0])\
             [:, np.newaxis]

y = np.array([236.4, 234.4, 252.8, 
              298.6, 314.2, 342.2, 
              360.8, 368.0, 391.2,
              390.8])

from sklearn.preprocessing import PolynomialFeatures

lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)
print(X_quad)
print(quadratic.get_feature_names())

temp = np.array([[2.0, 3.0],[3.0,4.0]])
quadratic2 = PolynomialFeatures(degree=2)
temp_quad = quadratic.fit_transform(temp)
print(temp_quad)
quadratic2.fit(temp) # fit_transform하면 에러
print(quadratic2.get_feature_names())