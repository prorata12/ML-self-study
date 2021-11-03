from default import *
# https://github.com/rickiepark/python-machine-learning-book-2nd-edition/raw/master/code/ch16/pg2265.txt
import numpy as np


"""
16.5 텐서플로로 글자 단위 언어 모델 구현
"""

"""
16.5.1 데이터 전처리
"""
## 텍스트를 읽고 처리합니다.
with open('./Chapter16/pg2265.txt', 'r', encoding='utf-8') as f: 
    text=f.read()

text = text[15858:] #법률 조항 제외
chars = set(text) #text에 들어있는 char 개수 세기
char2int = {ch:i for i,ch in enumerate(chars)} # 각 char를 int로 맵핑
int2char = dict(enumerate(chars))
text_ints = np.array([char2int[ch] for ch in text], 
                     dtype=np.int32)

len(text)   #162849
len(chars)  #65

# data x,y의 구조
"""
step1_1 step2 step3 | step1 step2 step3 | step1 step2 step3
step1_2
step1_3
...
step1_n
즉, 행이 한 스텝에서 들어갈 data 개수(batch_size)
그리고 step1~step3까지가 해당 sequence의 길이(num_step)
즉 | 를 기준으로 나뉜 구역이 배치 하나 하나에 해당(minibatch)

x의 다음 char가 y의 값에 해당(true prediction)
"""
def reshape_data(sequence, batch_size, num_steps):
    mini_batch_length = batch_size * num_steps # |로 나뉜 한 mini batch
    num_batches = int(len(sequence) / mini_batch_length) # 전체 sequence를 mini_batch 크기로 나눈게 batch 개수
    if num_batches*mini_batch_length + 1 > len(sequence): # 만약 sequence랑 동일하면, y 값이 하나 존재할 수 없으므로
        num_batches = num_batches - 1
    ## 전체 배치에 포함되지 않는 시퀀스 끝부분은 삭제합니다.
    x = sequence[0 : num_batches*mini_batch_length]
    y = sequence[1 : num_batches*mini_batch_length + 1]
    ## x와 y를 시퀀스 배치의 리스트로 나눕니다.
    x_batch_splits = np.split(x, batch_size) #batch_size가 행이 되도록 쭉 나눔
    y_batch_splits = np.split(y, batch_size)
    print('dim1',np.array(x_batch_splits).shape)
    ## 합쳐진 배치 크기는
    ## batch_size x mini_batch_length가 됩니다.
    x = np.stack(x_batch_splits)
    y = np.stack(y_batch_splits)
    print('dim2',np.array(x).shape)
    
    return x, y

train_x, train_y = reshape_data(text_ints, 64, 10)
print(train_x.shape)
print(train_x[0, :10])
print(train_y[0, :10])
print(''.join(int2char[i] for i in train_x[0, :10]))
print(''.join(int2char[i] for i in train_y[0, :10]))

def create_batch_generator(data_x, data_y, num_steps):
    batch_size, tot_batch_length = data_x.shape[0:2]   
    num_batches = int(tot_batch_length/num_steps)
    for b in range(num_batches):
        yield (data_x[:, b*num_steps: (b+1)*num_steps], 
               data_y[:, b*num_steps: (b+1)*num_steps])


bgen = create_batch_generator(train_x[:,:100], train_y[:,:100], 15) # setence sequence를 num_step개씩 나누어 자름 (즉, 글자 15개씩)
for x, y in bgen:
    print(x.shape, y.shape, end='  ')
    print(''.join(int2char[i] for i in x[0,:]).replace('\n', '*'), '    ',
          ''.join(int2char[i] for i in y[0,:]).replace('\n', '*'))

batch_size = 64
num_steps = 100 
train_x, train_y = reshape_data(text_ints, batch_size, num_steps)
print(train_x.shape, train_y.shape)

from tensorflow.keras.utils import to_categorical

train_encoded_x = to_categorical(train_x) #원핫인코딩, to_categorical은 값이 0부터 시작한다고 가정함 / 따라서 길이는 train_x의 최댓값 + 1
train_encoded_y = to_categorical(train_y)
print(train_encoded_x.shape, train_encoded_y.shape)


"""
16.5.2 RNN 모델 만들기
"""

char_model = models.Sequential()

num_classes = len(chars)

char_model.add(layers.LSTM(128, input_shape=(None, num_classes), 
                           return_sequences=True))
char_model.add(layers.TimeDistributed(layers.Dense(num_classes, 
                                                   activation='softmax')))
char_model.summary()

"""
16.5.3 RNN 모델 훈련
"""

from tensorflow.keras.optimizers import Adam

adam = Adam(clipnorm=5.0)
char_model.compile(loss='categorical_crossentropy', optimizer=adam)

#loaded, char_model = pickle_load('./Chapter16/char_model.h5', char_model)
if (not os.path.isfile('./Chapter16/char_rnn_checkpoint.h5')):
#if loaded == 0:
    #callback_list = [ModelCheckpoint(filepath='./Chapter16/char_rnn_checkpoint.h5')]
    callback_list = [ModelCheckpoint(filepath='./Chapter16/char_rnn_checkpoint.h5', 
                                 monitor='loss', 
                                 save_best_only=True)]

    for i in range(500):
        bgen = create_batch_generator(train_encoded_x, 
                                    train_encoded_y, num_steps)
        char_model.fit_generator(bgen, steps_per_epoch=25, epochs=1, 
                                callbacks=callback_list, verbose=0)                        
    #pickle_save('./Chapter16/char_model.h5', char_model)
else:
    char_model.load_weights('./Chapter16/char_rnn_checkpoint.h5')
    #loaded, char_model = pickle_load('./Chapter16/char_model.h5', char_model)


"""
16.5.4 글자 단위 RNN 모델로 텍스트 생성
"""
np.random.seed(42)

def get_top_char(probas, char_size, top_n=5):
    p = np.squeeze(probas)
    p[np.argsort(p)[:-top_n]] = 0.0
    p = p / np.sum(p)
    ch_id = np.random.choice(char_size, 1, p=p)[0]
    return ch_id


seed_text = "The "
for ch in seed_text:
    num = [char2int[ch]]
    onehot = to_categorical(num, num_classes=65)
    onehot = np.expand_dims(onehot, axis=0)
    probas = char_model.predict(onehot)
num = get_top_char(probas, len(chars))
seed_text += int2char[num]


for i in range(500):
    onehot = to_categorical([num], num_classes=65)
    onehot = np.expand_dims(onehot, axis=0)
    probas = char_model.predict(onehot)
    num = get_top_char(probas, len(chars))
    seed_text += int2char[num]
print(seed_text)