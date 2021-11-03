from default import *
os.system('cls')

import struct
plt.close('all')
# load_mnist in Chapter 13
def load_mnist(path, kind='train'):
    """`path`에서 MNIST 데이터 적재하기"""
    labels_path = os.path.join(path, 
                               '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, 
                               '%s-Images.idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
 
    return images, labels

# 28 x 28 images (gray)
X_data, y_data = load_mnist('./Chapter15', kind='train')
print('행: %d,  열: %d' %(X_data.shape[0], 
                                 X_data.shape[1]))

X_test, y_test = load_mnist('./Chapter15', kind='t10k')
print('행: %d,  열: %d' %(X_test.shape[0], 
                                 X_test.shape[1]))

X_train, y_train = X_data[:50000,:], y_data[:50000]
X_valid, y_valid = X_data[50000:,:], y_data[50000:]

print('훈련 세트: ', X_train.shape, y_train.shape)
print('검증 세트: ', X_valid.shape, y_valid.shape)
print('테스트 세트: ', X_test.shape, y_test.shape)


mean_vals = np.mean(X_train, axis=0) # 각 pixel마다 mean_vals 계산
print(f'{mean_vals.shape}')
std_val = np.std(X_train)
print(f'{std_val.shape}')
# std_val : pixel 전체의 std 계산 [특정 pixel은 항상 255(흰색)일수도 있기 때문에...]
# => 이미지에서는 한 pixel의 값이 항상 동일할 수 있으므로, std 값을 구할때 전체로 구해줘야한다.
# 그렇지않으면 std_val = 0 이 되어서 표준화 과정에서 error 발생할 수 있다.

# 이것보단 std_val이 0일때 따로 처리해주는게 나을까...? 어차피 centered = 0 으로 고정되니까 (X)
# => test set에선 mean_vals와 다른 값이 들어올 수 있음. 이 경우에 std_val로 나누어주지 않으면 혼자 튀는 값이 나올 수 있다

X_train_centered = (X_train - mean_vals)/std_val
X_valid_centered = (X_valid - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

# CNN에 넣으려면 각 image를 28 x 28 x 1 로 원본 이미지 파일과 동일하게 입력해야 한다.
# 지금 위의 것은 한 image가 784 x 1 로 1차원 flatten 된 상태
# * 여기서는 Gray여서 28 x 28 x 1로, 마지막 차원이 필요없어보이지만 CNN 자체가 channel 차원을 받아들여야 하므로
# 마지막 차원을 필수로 넣어주어야 한다.

X_train_centered = X_train_centered.reshape((-1, 28, 28, 1)) # -1에는 당연히 sample 개수가 들어갈 것
X_valid_centered = X_valid_centered.reshape((-1, 28, 28, 1))
X_test_centered = X_test_centered.reshape((-1, 28, 28, 1))

# 이제 label을 one-hot encoding 해주자
from tensorflow.keras.utils import to_categorical

y_train_onehot = to_categorical(y_train)
y_valid_onehot = to_categorical(y_valid)
y_test_onehot = to_categorical(y_test)

print(y_train[0], y_train_onehot[0]) #Label 5 => [0 0 0 0 0 1 0 0 0 0]로 encoding됨

print('훈련 세트: ', X_train_centered.shape, y_train_onehot.shape)
print('검증 세트: ', X_valid_centered.shape, y_valid_onehot.shape)
print('테스트 세트: ', X_test_centered.shape, y_test_onehot.shape)

"""
15.3.3 텐서플로 tf.kears API로 CNN 구성
"""
print('\n\n')

from tensorflow.keras import layers, models

model = models.Sequential()

# Convolution layer
model.add(layers.Conv2D(32, (5, 5), padding='valid', 
                        activation='relu', input_shape=(28, 28, 1)))
# 32 - 필터수, (5,5) : 필터크기, padding : valid(default), same, full
# strides : n or (n, m) [default = 1 = (1, 1)]
# kernel_initializer -> default = glorot_uniform
# bias_initializer -> default= zeros

# Pooling layer
model.add(layers.MaxPool2D((2, 2)))
# Pooling은 tuple로, argument 하나만을 넘겨줌
# strides -> default = None (겹치지 않게 설정), 보통 풀링에서는 지정하지 않음(default)
# (2,2) pooling이므로 크기가 절반으로 줄어듦음 # 단, feature map 개수는 유지
# Pooling은 가중치가 존재하지 않음!

model.add(layers.Conv2D(64, (5, 5), padding='valid', 
                        activation='relu'))
# 필터수 64개                        
# 파라미터 개수 : 5 x 5 x <32> 크기의 filter가 64개 존재. 5 x 5 x 32 x 64 = 51264
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5)) # Dropout을 여러 군데 넣는 경우, 입력층에 가까울수록 낮은 Drop out을 쓰는 것을 권장
model.add(layers.Dense(10, activation='softmax'))

print(model.summary())
print(dir(model))
print(model.weights)


model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['acc'])

import time
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import pickle

if not os.path.isfile('./Chapter15/cnn_checkpoint.h5'):
    ### point
    callback_list = [ModelCheckpoint(filepath='./Chapter15/cnn_checkpoint.h5', 
                                 monitor='val_loss', 
                                 save_best_only=True)]

    history = model.fit(X_train_centered, y_train_onehot, 
                        batch_size=64, epochs=20, 
                        validation_data=(X_valid_centered, y_valid_onehot),
                        callbacks=callback_list)
    ### pointend    
    #model.save('./Chapter14/sim_model.h5')
    with open('./Chapter15/history.pkl', 'wb') as history_best:
        pickle.dump(history.history, history_best)
else:
    with open('./Chapter15/history.pkl', 'rb') as history_best:
        history_history = pickle.load(history_best)
    #model.load_weights('./Chapter15/cnn_checkpoint.h5')
    model = tf.keras.models.load_model('./Chapter15/cnn_checkpoint.h5')

epochs = np.arange(1,21)
plt.plot(epochs, history_history['loss'])
plt.plot(epochs, history_history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


plt.plot(epochs, history_history['acc'])
plt.plot(epochs, history_history['val_acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

model.evaluate(X_test_centered, y_test_onehot)


print(np.argmax(model.predict(X_test_centered[:10]), axis=1)) #여기서는 label index와 label이 동일하지만, 일반적으로는 다르다는 것을 유의!
print(y_test[:10])
#predict -> 각 label이 나올 확률을 return.
#argmax를 통해 그중 maximum값을 찾음 : axis=1로 해서 feature 방향중에 max를 찾음(각 row마다 max찾음)

fig = plt.figure(figsize=(20, 5)) #전체 수, 한 행에 들어갈 수
for i in range(10):
    fig.add_subplot(4, 5, i+1) #행, 열, 번호
    plt.imshow(X_test_centered[i].reshape(28, 28))

for i in range(10):
    fig.add_subplot(4, 5, i+11)
    plt.imshow(X_test[i].reshape(28, 28))

plt.tight_layout()
plt.show()

"""
15.3.5 활성화 출력과 필터 시각화
"""
#첫번째 층(Conv) 의 output 출력해보기

first_layer = model.layers[0]
print(first_layer)
print(model.input) # None => 배치 크기 차원


first_activation = models.Model(inputs=model.input, 
                                outputs=first_layer.output) # layer 자체를 받아오는 방법
activation = first_activation.predict(X_test_centered[:10]) # model의 가중치도 같이 받아오므로, 추가 학습 필요 없음


print(activation.shape) # (10,24,24,32) => sample 수, 24*24(conv), 32(filter 개수)

#첫번째 sample(7)의 filter 결과 출력해보기
fig = plt.figure(figsize=(10, 15))
for i in range(32):
    fig.add_subplot(7, 5, i+1)
    plt.imshow(activation[0, :, :, i])
plt.show()
#세번재 sample(0)의 filter 결과 출력해보기
fig = plt.figure(figsize=(10, 15))
for i in range(32):
    fig.add_subplot(7, 5, i+1)
    plt.imshow(activation[3, :, :, i])
plt.show()

fig = plt.figure(figsize=(10, 15))
for i in range(32):
    fig.add_subplot(7, 5, i+1)
    plt.imshow(first_layer.kernel[:, :, 0, i])
#filter 정보는 kernel에 저장
print(first_layer.kernel.shape)
plt.show()