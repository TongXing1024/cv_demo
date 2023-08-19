# Author : 佟星TONGXING
# Date : 2022/11/13 20:03
# Version: 1.0
import random

"""
求1 -100的和
"""
def getSum1():
    i = 1
    sum = 0
    while i<=100:
        sum = sum+i
        i+=1
    print(sum)











def getSum():
    i = 1
    sum = 0
    while i <= 100:
        sum = sum + i
        i += 1
    print(f"1-100的和为:{sum}")


#设置一个范围1-100的随机整数变量,通过while循环,配合input语句,判断输入的数字是否等于随机数
def getRandom():
    num = random.randint(1, 100)
    i = int(input("请输入一个数字:"))
    while True:
        if i == num:
            print("恭喜你,猜对了!")
            break  #终止循环
        elif i<num:
            i = int(input("有点小,请再次输入:"))
        else:
            i = int(input("有点大,请再次输入:"))


if __name__ == '__main__':
   getSum1()