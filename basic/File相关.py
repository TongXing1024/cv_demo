# Author : 佟星TONGXING
# Date : 2022/11/16 9:22
# Version: 1.0
"""
read(num):读取num个字符,如果没有传入num,那么表示读取文件中所有数据
readline():一次读取一行
readlines():按照行的方式把文件一次性读取,返回一个列表,每一行的数据作为列表中的一个元素
"""


def file_read():
    f = open("D:/File/test.txt", 'r', encoding='utf-8')
    # with open 语法
    # 自动关闭文件
    with open("D://File/test1.txt", "r", encoding="utf-8") as f:
        for line in f:
            print(line)

def file_write():
    """
    文件写入
    :return:
    """
    f = open("D://File/test1.txt", "w", encoding="utf-8")
    f.write("hello world")
    f.close()

def file_append():
    """
    文件的追加
    :return:
    """
    f = open("D://File/test2.txt", "a", encoding="utf-8")
    f.write("汗滴禾下土")
    f.close()

def file_copy():
    """
    文件拷贝
    :return:
    """
    f1 = open("D://File/test.txt", "r", encoding="utf-8")
    f2 = open("D://File/test1.txt", "a", encoding="utf-8")
    for line in f1:
        f2.write(line)
    f2.close()
    f1.close()


if __name__ == '__main__':
    file_copy()
