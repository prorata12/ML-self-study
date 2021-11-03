"""
14. 텐서플로의 구조 자세히 알아보기 (14.7)
14.7 - 계산 그래프 시각화
"""
import os
from default import *

# Random Data 재생성

np.random.seed(0)

def make_random_data():
    x = np.random.uniform(low=-2, high=2, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0, #center
                             scale=(0.5 + t*t/3), #std
                             size=None)
        y.append(r)
    return  x, 1.726*x -0.84 + np.array(y)


x, y = make_random_data() 

x_train, y_train = x[:150], y[:150]
x_test, y_test = x[150:], y[150:]



#======= Tensor Board =======#
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=1))

callback_list = [tf.keras.callbacks.TensorBoard(log_dir='logs')]
model.compile(optimizer='sgd', loss='mse')
history = model.fit(x_train, y_train, epochs=300, 
                    callbacks=callback_list, validation_split=0.3, verbose = 0)
"""
#import pydotplus

input = tf.keras.Input(shape=(1,), dtype=tf.float32, batch_size=5)
hidden = tf.keras.layers.Dense(100)(input)
output = tf.keras.layers.Dense(1)(hidden)

model = tf.keras.Model(input, output)
model.compile(optimizer='sgd',loss='mse')
history = model.fit(x_train, y_train, epochs = 10, validation_split=0.2, verbose=0)
tf.keras.utils.plot_model(model, to_file='./Chapter14/model_1.png')
tf.keras.utils.plot_model(model, show_shapes=True, to_file='./Chapter14/model_2.png')

print(model.evaluate(x_test, y_test))

x_arr = np.arange(-2, 2, 0.1)
y_arr = model.predict(x_arr)

plt.figure()
plt.plot(x_train, y_train, 'bo')
plt.plot(x_test, y_test, 'bo', alpha=0.3)
plt.plot(x_arr, y_arr, '-r', lw=3)
plt.show()

print(dir(history))
print(dir(history.model))

