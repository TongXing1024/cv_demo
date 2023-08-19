# Author : 佟星TONGXING
# Date : 2022/11/10 22:34
# Version: 1.0


"""
等价于
a = 0
b = 1
"""
a,b = 0,1
while b<10:
    print(b)
    a,b = b,a+b
