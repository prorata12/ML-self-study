import tensorflow as tf

#(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

import struct, os
import numpy as np
 
# data 불러온 뒤 전처리

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

X_train, y_train = load_mnist('./Chapter13', kind='train')
print('행: %d,  열: %d' %(X_train.shape[0], 
                                 X_train.shape[1]))
X_test, y_test = load_mnist('./Chapter13', kind='t10k')
print('행: %d,  열: %d' %(X_test.shape[0], 
                                 X_test.shape[1]))

# Standardization
## 평균을 0으로 만들고 표준 편차로 나눕니다.
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val #??
 
del X_train, X_test
 
print(X_train_centered.shape, y_train.shape)

print(X_test_centered.shape, y_test.shape)

#one-hot-encoding
np.random.seed(123)

y_train_onehot = tf.keras.utils.to_categorical(y_train) #can use num_classes => input보다 더 큰 범위의 인코딩도 가능
 
print('처음 3개 레이블: ', y_train[:3])
print('\n처음 3개 레이블 (원-핫):\n', y_train_onehot[:3])


#13.2.2 Network 구성
model = tf.keras.models.Sequential()

model.add(
    tf.keras.layers.Dense(
        units=50,    #(1) - 절편은 미포함
        input_dim=X_train_centered.shape[1], #feature dim과 동일해야함
        kernel_initializer='glorot_uniform', #가중치 초기화, Xavier 초기화(이전/다음노드수고려) : 절편은 보통 0 으로 초기화
        bias_initializer='zeros',
        activation='tanh'))

model.add(
    tf.keras.layers.Dense(
        units=50,    #(2)
        input_dim=50, #(1)과 동일
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))

model.add(
    tf.keras.layers.Dense(
        units=y_train_onehot.shape[1],    #최종 output labels수
        input_dim=50, #(2)와 동일
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='softmax'))

print(model.summary())

#13.2.3
#모델 컴파일, Loss function, 최적화에 쓸 Optimizer(SGD)
#학습률 감소 ㅣ상수(점차감소) 및 momentum:이전 gradient만큼 다음 gradient를 보정
#cost(loss) function - crossentrophy ~ Logistic loss func
#softmax는 다중 클래스에서의 logistic loss func

sgd_optimizer = tf.keras.optimizers.SGD(
    lr=0.001, decay=1e-7, momentum=.9)
#lr - eta #decay만큼 반복시 학습률 절반감소
model.compile(optimizer=sgd_optimizer,
              loss='categorical_crossentropy')

history = model.fit(X_train_centered, y_train_onehot,
    batch_size=64, epochs=50,
    verbose=2, #0:silent 1:progress bar 2:oneline/epoch
    validation_split=0.1) #10%를 validation set으로 떼어냄

y_train_pred = model.predict_classes(X_train_centered, verbose=0)
print('처음 3개 예측: ', y_train_pred[:3])


#train set accuracy
y_train_pred = model.predict_classes(X_train_centered, 
                                     verbose=0)
correct_preds = np.sum(y_train == y_train_pred, axis=0) 
train_acc = correct_preds / y_train.shape[0]

print('처음 3개 예측: ', y_train_pred[:3])
print('훈련 정확도: %.2f%%' % (train_acc * 100))

#test set accuracy
y_test_pred = model.predict_classes(X_test_centered, 
                                    verbose=0)

correct_preds = np.sum(y_test == y_test_pred, axis=0) 
test_acc = correct_preds / y_test.shape[0]
print('테스트 정확도: %.2f%%' % (test_acc * 100))