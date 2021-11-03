from default import *
os.system('cls')

"""
15.1 합성곱 신경망(Convolutional Neural Network, CNN)의 구성 요소
- 이미지 분류에 탁월한 성능
"""

# 15.1.1. CNN과 특성 계층 학습
"""
핵심 Feature를 추출하는 것은 몹시 중요
Neural Network => 중요한 Feature가 무엇인지 학습할 수 있다
: 신경망이 특성 추출 엔진으로 생각되기도 하는 이유
Multi Layer Neural Network, Deep CNN은 low-level feature(edge, circle 등)으로부터 high-leve feature(dog,cat)을 형성한다
: 앞쪽 층이 low-level, 뒤쪽 층이 high-level
"""

"""
Feature map : 원본 Image의 5x5(or nxn) pixel patch로부터 값을 유도한 테이블.
이때 nxn pixel patch를 local receptive field(국부 수용장) 이라고 부름
"""

"""
CNN의 중요한 특성
(1) Sparse Connectivity(희소 연결) : Feature map의 한 element는 원본 이미지의 일부(patch)에만 대응됨.
    => 즉 각 element가 image 전체가 아닌 local receptive field(patch)에만 대응.
(2) Parameter Sharing(파라미터 공유) : 모든 patch는 동일한 가중치(weight)를 공유함.
    => 가중치(weight,parameter)의 개수가 몹시 줄어듦 => Important Feature를 더 잘 잡아내게 됨
    => 가까이 있는 pixel들이 당연히 먼 pixel들 보다 연관성 높음
"""
"""
여러개의 합성곱(Conv)층, 풀링(Pooling)층으로 구성
=> 풀링층엔 가중치/절편 유닛(parameter)이 존재하지 않음
"""

#실제 행해지는 필터링 =>위노그라드 최솟값 필터링 (Winograd's Minimal Filtering)

"""
15.1.3 서브샘플링
"""
# 두 종류의 풀링 - 최대 풀링, 평균 풀링(Max-pooling, mean/average pooling)
# max-pooling -> 지역 불변성 :  국부적 변화가 결과를 바꾸지 못함 => 잡음에 안정적,
# pooling => feature 개수 줄임 => 효율성 증가, 과대적합 방지
# 일반적으로 pooling은 겹치지 않음 (P_n1*n2 크기의 풀링은 (n1,n2)의 stride를 가지게 함)
# 겹침 풀링을 사용하는 경우도 존재


