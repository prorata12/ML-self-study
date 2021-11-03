"""
14. 텐서플로의 구조 자세히 알아보기 (14.4 ~ 14.5)
14.4 - tf1 <-> tf2의 차이 (graph 관점)
14.5 - tf.Variable (.numpy() = np type)
    - tf.Variable.assign(value)


"""
from default import *

# python - @ decorator
# closure - function call을 할때, function이 가진 environment를 저장하는 곳? 정도

"""
@func_name
def new_func():
    contents
    return
위와 같이 작성시, new_func = func_name(new_func) 와 같이 들어감.
new_func = wrapper_function(x,y)

즉, decorator function의 arguments로 function을 넘겨주고, 그걸 새로운 function의 정의로 바꾸는 것.
decorator는 wrapper라고 보면 됨. (기존 new_func에 추가적인 기능을 넣기 위해 decorate)
"""


# session - 자신의 resource를 다른 user에게 사용하도록 허락하는 기간 (by log-in)
#         - Session의 사용을 통해 다른 user로부터 보호 & 비정상적 log-out에서의 data 보호



"""
14.5 - 텐서플로의 변수
"""
# variable in tf => 훈련과정동안 업데이트할 값들 (가중치(weight) 등)
# 초기값 필요
# https://www.tensorflow.org/guide/variables


# 정의 방법 : tf.Varialbe(initial_value, name='name_optional')
# shape, dtype 설정 불가. initial_value의 shape/dtype으로 고정

# tf1의 get_variable()은 tensorflow.compat.v1 으로 이동함
# tf1에서 variable은 graph node로 존재. 실제 파이썬 객체가 아님 => variable scope 컨트롤이 어려움
# tf2에서는 일반 파이썬 객체처럼 변수 공유 가능



# tf1 version - graph 생성
g1 = tf.Graph()

with g1.as_default():
    w1 = tf.Variable(np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8]]), name='w1')
print(w1) #w1에는 실제 값이 포함되어있지 않음. 사용 전에 초기화 필요함

print(g1.get_operations()) # 사용 전에 초기화해야하는 변수들

with g1.as_default():
    init = tf.compat.v1.global_variables_initializer() # 위의 변수들을 모두 초기화해주는 함수
    print(init.node_def)



#===== tf에서 variable 값 바꾸기 (assign) =====#
# tf1 - not changed
with g1.as_default():
    w1 = w1 + 1 # 이와 같이 w1 + 1으로 입력하면, update되는게 아니라 w1+1 값을 리턴할 뿐. 즉, w1는 바뀌지 않는다
    print(w1)
    
with tf.compat.v1.Session(graph=g1) as sess:
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    print(sess.run(w1))
    print(sess.run(w1))

# tf1 - changed
g2 = tf.Graph()

with g2.as_default():
    w1 = tf.Variable(np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8]]), name='w1')
    w1 = w1.assign(w1 + 1) #이렇게 assign해주면 그제서야 값이 바뀐다

with tf.compat.v1.Session(graph=g2) as sess:
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    print(sess.run(w1))
    print(sess.run(w1))


#tf2 - changed
w2 = tf.Variable(np.array([[1, 2, 3, 4],
                          [5, 6, 7, 8]]), name='w2')
printing(w2)
print(w2 + 1) # 이렇게 +1을 하면, tf2에서는 자동으로 tensor로 return 해준다.

#즉, 만약 w2를 바꾸고 싶으면 assign을 call해주면 된다.
w2.assign(w2+1)
printing(w2) #tf.Varialbe

w2numpy = w2.numpy() 
printing(w2numpy) #numpy.ndarray (= np.array([]))
print(type(w2.numpy()))
