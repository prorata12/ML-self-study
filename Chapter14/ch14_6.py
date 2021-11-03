"""
14. 텐서플로의 구조 자세히 알아보기 (14.6)
14.6 - tf.keras API 사용하기
    - model 저장 및 불러오기
    - history를 통해 acc/loss 어떻게 변하는지 확인하기
"""
import os
from default import *




#===== 랜덤 데이터 생성 =====#
os.system('cls')
print('#===== 랜덤 데이터 생성 =====#')
## 랜덤한 회귀용 예제 데이터셋을 만듭니다

np.random.seed(0)

def make_random_data():
    x = np.random.uniform(low=-2, high=2, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0, #centor
                             scale=(0.5 + t*t/3), #std
                             size=None)
        y.append(r)
    return  x, 1.726*x -0.84 + np.array(y)


x, y = make_random_data() 

# value = np.linspace(-2,2,200)

# plt.subplot(3,1,1)
# plt.plot(x, np.zeros(shape=(200,)), 'o')

# plt.subplot(3,1,2)
# plt.plot(y, np.zeros(shape=(200,)), 'o')

# plt.subplot(3,1,3)
# plt.plot(value, y, 'o')


plt.plot(x,y,'o') #x=0를 중심으로, 멀어질수록 오차가 커지는 1.726x - 0.84 그래프
plt.show()





#===== 학습 과정 =====#
os.system('cls')
print('#===== 학습 과정 =====#')
x_train, y_train = x[:150], y[:150]
x_test, y_test = x[150:], y[150:]


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=1)) #units : output dim

import pickle
model.compile(optimizer='sgd', loss='mse')

#save the model if file doesn't exist
if not os.path.isfile('./Chapter14/Sequential.h5'):
    history = model.fit(x_train, y_train, epochs=300, 
                        validation_split=0.3, verbose=2)
    print(type(history))
    with open('./Chapter14/history.pkl', 'wb') as history_file:
        pickle.dump(history.history, history_file)   
    model.save_weights('./Chapter14/Sequential.h5')
    #pickle.dump(history, open(os.path.abspath('history.pkl'), 'wb'), protocol=4)   

else:
    model.load_weights('./Chapter14/Sequential.h5')
    with open('./Chapter14/history.pkl', 'rb') as history_file:
        history_history = pickle.load(history_file)

"""
loss : 훈련 손실값
acc : 훈련 정확도
val_loss : 검증 손실값
val_acc : 검증 정확도
"""
epochs = np.arange(1, 300+1)

#if save file doesn't exist....
try:
    history
    plt.plot(epochs, history.history['loss'], label='Training loss')
#if save file exists....
except NameError:
    plt.plot(epochs, history_history['loss'], label='Training loss')


plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
print(model.summary())
print('layers=',model._layers) #input layer, dense layer 존재 [sequential model의 summary에서는 input layer를 생략함]
plt.show()

plt.plot(x,y,'o')
plt.plot(x_train,model.predict(x_train))
plt.title('data and prediction')
plt.show()

"""
14.6.2 함수형 API - 복잡한 network를 만드는데 도움
"""
os.system('cls')
print('#===== 14.6.2. 함수형 API =====#')

input = tf.keras.Input(shape=(1,)) # 1-dim(feature) input, Input instance 생성
output = tf.keras.layers.Dense(1)(input) # Dense class를 함수처럼 호출 (python 특유의 __call__ method (in class) 이용)

# dense = tf.keras.layers.Dense(1) # 위의 코드를 => 이렇게 정의하면 가중치를 공유하여 네트워크의 다른 곳에 사용 가능
# output = dense(input) or output = dense.__call__(input) # 와 같이 사용 가능



model = tf.keras.Model(input, output)

print(model.summary()) # input layers는 input을 입력하기 위한 층. 다른 처리를 하지 않음 (학습 x)
# sequential model에서도 사실 input layer를 자동으로 출력해줌. ._layers에서 확인가능


#모델 학습은 동일
model.compile(optimizer='sgd', loss='mse')
# history = model.fit(x_train, y_train, epochs=300, 
#                     validation_split=0.3, verbose=0)

#save the model if file doesn't exist
if not os.path.isfile('./Chapter14/function.h5'):
    history = model.fit(x_train, y_train, epochs=300, 
                        validation_split=0.3, verbose=2)
    print(type(history))
    with open('./Chapter14/history2.pkl', 'wb') as history_file:
        pickle.dump(history.history, history_file)   
    model.save_weights('./Chapter14/function.h5')
    #pickle.dump(history, open(os.path.abspath('history.pkl'), 'wb'), protocol=4)   

else:
    model.load_weights('./Chapter14/function.h5')
    with open('./Chapter14/history2.pkl', 'rb') as history_file:
        history_history = pickle.load(history_file)

try:
    history
    plt.plot(epochs, history.history['loss'], label='Training loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation loss')
except NameError:
    plt.plot(epochs, history_history['loss'], label='Training loss')
    plt.plot(epochs, history_history['val_loss'], label='Validation loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""
14.6.3. tf.keras의 모델 저장 및 복원 - 위에서 구현 
"""
os.system('cls')
print('#===== 14.6.3. tf.keras의 모델 저장 및 복원 =====#')

# model.save_weights('./Chapter14/simple_weights.h5')

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(units=1, input_dim=1))
# model.compile(optimizer='sgd', loss='mse')
# model.load_weights('./Chapter14/simple_weights.h5')

# 복원 확인
model.evaluate(x_test, y_test)

#model 전체 복원 (network 구조 추가 필요 없음)
model.save('simple_model.h5')

model = tf.keras.models.load_model('simple_model.h5')
model.evaluate(x_test,y_test)


#===== best model만 저장하기 =====#
if not os.path.isfile('./Chapter14/my_model.h5'):
    ### point
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_dim=1))
    model.compile(optimizer='sgd',loss='mse')    
    callback_list = [tf.keras.callbacks.ModelCheckpoint(filepath='./Chapter14/my_model.h5', 
                                    monitor='val_loss', save_best_only=True), #validation loss를 기준으로 best 책정 / save_best_only로 best만 저장할지 체크
                    tf.keras.callbacks.EarlyStopping(patience=5)] #성능개선이 되지 않을때 멈춤
    history = model.fit(x_train, y_train, epochs=300, 
                        validation_split=0.2, callbacks=callback_list)
    ### pointend    
    model.save('./Chapter14/sim_model.h5')
    with open('./Chapter14/history3.pkl', 'wb') as history_best:
        pickle.dump(history.history, history_best)
else:
    with open('./Chapter14/history3.pkl', 'rb') as history_best:
        history_history = pickle.load(history_best)
    model = tf.keras.models.load_model('./Chapter14/sim_model.h5')

try:
    history
    ### point
    epochs = np.arange(1, len(history.history['loss'])+1)
    plt.plot(epochs, history.history['loss'], label='Training loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation loss')
    ### pointend
except:
    epochs = np.arange(1, len(history_history['loss'])+1)
    plt.plot(epochs, history_history['loss'], label='Training loss')
    plt.plot(epochs, history_history['val_loss'], label='Validation loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#model = tf.keras.models.load_model('simple_model.h5')
#model.load_weights('my_model.h5')
print('loaded model')
model.evaluate(x_test, y_test)

x_arr = np.arange(-2, 2, 0.1)
y_arr = model.predict(x_arr)

plt.figure()
plt.plot(x_train, y_train, 'bo')
plt.plot(x_test, y_test, 'bo', alpha=0.3)
plt.plot(x_arr, y_arr, '-r', lw=3)
plt.show()