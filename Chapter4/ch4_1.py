import pandas as pd

from io import StringIO
import os
os.system('cls')

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)
"""
4.1.1 check if data contain null
"""
print(df.isnull()) # Return : True/False
print(df.isnull().sum()) # Return : num of True in a column

#sklearn도 np array 처리 가능하게 개발
#그러나 pd의 df 이용하여 전처리하는게 종종 더 편함
#df.values하면 np array 얻을 수 있음


"""
4.1.2 delete row/col with null value
"""

print(df.dropna(axis=0)) # 행(sample)을 기준으로, null이 하나라도 있으면 drop
print(df.dropna(axis=1)) # 열(feature)을 기준으로, null이 하나라도 있으면 drop
print(df.dropna(how='all')) # 열(feature) 전체가 null인 경우에만 drop
print(df.dropna(thresh=4)) # 행(sample)에서 feature 값이 4개 미만으로 존재하는 행을 drop #not necessarily number
print(df.dropna(subset=['C'])) # C열에서 NaN이 있는 행만 삭제

"""
4.1.3 replacement of null
"""


