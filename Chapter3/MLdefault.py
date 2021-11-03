# 내장 함수인 print를 print_temp에 저장
#print_temp = print

# #print 함수를 덮어씀
# def print(*args):
#     global print
#     return print(*args)

#덮어쓸 print 함수를 만듦(아무것도 출력하지 않는 n개의 arg를 받는 함수)
def print_none(*args):
    return 0

#pset으로 MLdefault의 print함수를 컨트롤
def pset(x):
    if x == 0 :
        return print_none
    return print