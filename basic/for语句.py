# Author : 佟星TONGXING
# Date : 2022/11/13 20:23
# Version: 1.0
"""
range语句:
语法一:range(num)  获取一个从0开始,到num结束的一个数字序列(不含num本身)
语法一:range(num1,num2)  获取一个从num1开始,到num2结束的一个数字序列(不含num2本身)
语法一:range(num1,num2,step)  获取一个从num1开始,到num2结束的一个数字序列(不含num2本身),step表示数字之间的步长,默认为1
"""


def getANum(name: str):
    """
    查找一个字符串中含有多少个'a'
    :param name:
    :return:
    """
    count = 0
    for ele in name:
        if ele == 'a':
            count += 1
    return count



def printTable():
    """
    打印99乘法表
    :return:
    """
    #外层控制行数
    for i in range(1, 10):
        #内层循环控制每一行的内容
        for j in range(1,i+1):
            print(f"{j} * {i} = {i * j}\t",end="")
        print()

if __name__ == '__main__':
    count = getANum("itcast is a brand of itcast")
    print(f"含有A的个数有:{count}")
    printTable()