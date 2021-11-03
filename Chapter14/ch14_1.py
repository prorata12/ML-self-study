"""
14. 텐서플로의 구조 자세히 알아보기 (14.1 ~ 14.3)
14.2 - Tensor의 rank, shape
14.3 - tf.constant <-> numpy.array
        tf.reshape, tf.transpose(perm), tf.split, tf.concat

14.4 - tf1 <-> tf2의 차이 (graph 관점)
14.5 - tf.Variable (.numpy() = np type)
    - tf.Variable.assign(value)

"""
#===== Tensorflow 사용시 발생하는 메시지 숨기기 =====#
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from default import *


"""
14.2. Rank of Tensor
"""
#===== Tensorflow의 차원 구하는 방법(rank, shape) =====#

# scalar - 0차원 텐서 (rank-0 tensor)
# vector - 1차원 텐서 (rank-1 tensor)
# matrix(행렬) - 2차원 텐서 (rank-2 tensor)
import tensorflow as tf
import numpy as np
print("14.2")
t1 = tf.constant(np.pi) #rank-0 , size = ()
t2 = tf.constant([1,2,3,4]) #rank-1, size = (4,) # [[]] 꼴이어야 (4,1)
t3 = tf.constant([[1,2],[3,4]]) #rank-2, size = (2,2)

#get rank of tensor
r1 = tf.rank(t1)
r2 = tf.rank(t2)
r3 = tf.rank(t3)

## 크기를 구합니다
s1 = t1.get_shape()
s2 = t2.get_shape()
s3 = t3.get_shape()
print('크기:', s1, s2, s3)

print('랭크:', 
      r1.numpy(), 
      r2.numpy(), 
      r3.numpy())



"""
14.3. Tensor --> numpy.array
"""
#===== Tensorflow constant와 numpy arr의 호환성 =====#

# tensor => tf.constant(value).get_shape()  # .shape도 작동함
# numpy => np.array(value).shape

print("\n\n14.3")
arr = np.array([[1., 2., 3., 3.5],
                [4., 5., 6., 6.5],
                [7., 8., 9., 9.5]])
T1 = tf.constant(arr) #np.array를 바로 tf.constant(텐서)로 변환가능
print(T1)
s = T1.get_shape()

#아래 두개의 출력값 동일!! [ .get_shape() = .shape , attribute로 존재]
print('T1의 크기:', s)
print('T1의 크기:', T1.shape)


T2 = tf.Variable(np.random.normal(size=s)) #get_shape로 얻은 (3,4)꼴의 input을 size로 넣어줌. np에서 제공하는 기능
print('<Value of T2>\n',T2)
T3 = tf.Variable(np.random.normal(size=s[0])) #size=3  -> (3,)과 동일!
print('<Value of T3>\n',T3)
T3_2 = tf.Variable(np.random.normal(size=(3,))) #size=(3,)
print('<Value of T3_2>\n',T3_2)

# value of T1
#[[1.  2.  3.  3.5]
# [4.  5.  6.  6.5]
# [7.  8.  9.  9.5]], shape=(3, 4), dtype=float64)

"""
Tensorflow를 이용한 행렬 변형 (reshape, transpose(with perm), split, concat)
"""
#===== tensorflow reshape =====#

T4 = tf.reshape(T1, shape=[1, 1, -1])
print('<Value of T4>\n',T4)
T5 = tf.reshape(T1, shape=[1, 3, -1])
print('<Value of T5>\n',T5)

# Transpose of Matrix
# Numpy에서는...
# arr.T     arr.transpose()     np.transpose(arr)
# TensorFlow에서는...
# tf.transpose(arr)



#===== tensorflow split =====#
T_temp = tf.transpose(T1)
print('<Value of T_temp>\n',T_temp)

#T5.shape = (1,3,4)
T6 = tf.transpose(T5, perm=[2, 1, 0]) # perm => 0,1,2로 입력해야함. 각각이 0-dim 1-dim 2-dim
print('<Value of T6>\n',T6)
print('<Dim of T6>\n',T6.shape)
#T6.shape = (4,3,1)
T7 = tf.transpose(T5, perm=[0, 2, 1])
print('<Value of T7>\n',T7)
#T6.shape = (1,4,3)



#===== tensorflow split =====#

t5_splt = tf.split(T5, 
                   num_or_size_splits=2, 
                   axis=2)
# 결과는 tf tensor의 list!

printing(t5_splt) #dimension : (1,3,4) => (1,3,2)

# t5_split_2 = tf.split(T5, 
#                    num_or_size_splits=2, 
#                    axis=1)
# 불가능 => 열이 3개로, 2의 배수가 아니기 때문
t5_splt_2 = tf.split(T5, 
                   num_or_size_splits=3, 
                   axis=1)
printing(t5_splt_2) #dimension : (1, 3, 4) => (1, 1, 4)
t5_splt_2_0 = t5_splt_2[0]
print(t5_splt_2_0)



#===== tensorflow concat =====#

t1 = tf.ones(shape=(5, 1), dtype=tf.float32)
t2 = tf.zeros(shape=(5, 1), dtype=tf.float32)
printing(t1)
printing(t2)

t3 = tf.concat([t1, t2], axis=0) #행방향으로 한칸씩 진행하면서 합성(하나끝나서 진행 불가능하면 그제서야 다음꺼 concat!)
printing(t3) #즉, 행방향으로 array가 길어짐
print(t3.shape)
t4 = tf.concat([t1, t2], axis=1) #열방향으로 한칸씩 진행하면서 합성(하나끝나서 진행 불가능하면 그제서야 다음꺼 concat!)
printing(t4) #즉, 열방향으로 array가 길어짐
print(t4.shape)
