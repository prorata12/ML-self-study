"""
13.텐서플로를 이용하여 신경망 훈련

13.2 - data 전처리, keras를 이용한 망 구성, 훈련 및 예측
13.3 - logistic/softmax function 값 확인
"""

import tensorflow as tf
# Image - Adam Optimizer
#tesnorflow 1.x Version
## 그래프를 생성합니다
"""
g = tf.Graph()
with g.as_default():
    #placeholder -> 입력데이터(x or y, tuning이 필요한 파라미터들)
    x = tf.compat.v1.placeholder(dtype=tf.float32,
                       shape=(None), name='x')#shape:입력데이터크기
    w = tf.Variable(2.0, name='weight')
    b = tf.Variable(0.7, name='bias')

    z = w*x + b
    init = tf.compat.v1.global_variables_initializer()

## 세션을 만들고 그래프 g를 전달합니다
with tf.compat.v1.Session(graph=g) as sess:
    ## w와 b를 초기화합니다.
    sess.run(init)
    ## z를 평가합니다.
    for t in [1.0, 0.6, -1.8]:
        print('x=%4.1f --> z=%4.1f'%(
              t, sess.run(z, feed_dict={x:t})))

with tf.compat.v1.Session(graph=g) as sess:
    sess.run(init)
    print(sess.run(z, feed_dict={x:[1., 2., 3.]}))
print(z)
"""

#Tensorflow 2.x Version

# TF 2.0
w = tf.Variable(2.0, name='weight')
b = tf.Variable(0.7, name='bias')

# z를 평가합니다.
for x in [1.0, 0.6, -1.8]:
    z = w * x + b
    print('x=%4.1f --> z=%4.1f'%(x, z))
print(z) #마지막 값을 저장

z = w * [1., 2., 3.] + b
print(z.numpy())


