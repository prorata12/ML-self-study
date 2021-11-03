# -*- coding: utf-8 -*-
# decorator 관련 사이트
# http://schoolofweb.net/blog/posts/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%8D%B0%EC%BD%94%EB%A0%88%EC%9D%B4%ED%84%B0-decorator/

# === class decorator 전까지 읽음 === #

def decorator_function(original_function):
    def wrapper_function(*args, **kwargs):  #1 #args-> 일반 args, kwargs -> keyward args : dict형태를 받ㅇ므
        #print '{} 함수가 호출되기전 입니다.'.format(original_function.__name__)
        print(f'{original_function.__name__} 함수가 호출되기 전 입니다.')
        print(f'args={args}, kwargs={kwargs}')
        return original_function(*args, **kwargs)  #2 # wrapper function에서 ㅁargs 받기
    return wrapper_function


@decorator_function
def display():
    print('display 함수가 실행됐습니다.')


@decorator_function
def display_info(name, age, **kwargs):
    #print 'display_info({}, {}) 함수가 실행됐습니다.'.format(name, age)
    print(f'display_info({name}, {age}) 함수가 실행됐습니다.')
if __name__ == '__main__':
    display()
    print
    display_info('John', 25, a='key')