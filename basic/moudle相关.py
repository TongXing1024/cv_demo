# Author : 佟星TONGXING
# Date : 2022/11/16 9:49
# Version: 1.0

"""
模块就是一个python文件,里面有类,函数,变量等,我们可以拿过来用

[from 模块名] import [模块 | 类 | 变量 | 函数 | *] [as 别名]
常用的组合形式
    1. import 模块名
    2. from 模块名 import 类、变量、方法等
    3. from 模块名 import *
    4. from 模块名 as 别名
    5. from 模块名 import 功能名 as 别名
"""
import time #导入时间模块

def moudle_test():
    print("1111")
    time.sleep(5)
    print("22222")


if __name__ == '__main__':
    moudle_test()