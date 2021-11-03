"""
16. 순환 신경망으로 시퀀스 데이터 모델링
RNN(시퀀스 모델링) - 언어 번역, 이미지 캡셔닝, 텍스트 생성 등
시퀀스 모델릴 처리 방법 [ input -> output ]
: 다대일(many-to-one) : sequence -> vector (e.g. 감성분석[text->label])
: 일대다(one-to-many) : vector -> sequence (e.g. 이미지 캡셔닝[image->sentence])
: 다대다(many-to-many) : sequence -> sequence (e.g. 비디오 분류(각 프레임이 레이블링 - 동기적))
                                              (e.g. 언어 번역(한 언어를 모두 읽은 뒤 번역 - 비동기적))
"""
# 시계열 데이터 (time series data) - 시간 순서대로 나열된 data
# RNN의 경우, Loss function의 Gradient 계산시 W_hh^n 꼴로 먼 활성화 출력이 커지거나 작아짐 (Exploding/Vanishing Gradient)
# 때문에 |W_hh| = 1로 맞추거나, T-BPTT or LSTM을 사용
# T-BPTT : 먼 time step의 Gradient를 버림
# LSTM - Long Short-Term memory : 과거 data를 적절히 가져옴


# LSTM - 복잡도가 높아서 학습에 시간이 걸림
# GRU(Gated Recurrent Unit) - LSTM을 간략화한 버전, 계산 효율성이 높음
