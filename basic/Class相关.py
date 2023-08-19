# Author : 佟星TONGXING
# Date : 2022/11/16 10:45
# Version: 1.0


class Student:
    name = None
    gender = None
    nationality = None
    native = None
    age = None
    __id = None #私有成员变量 以__开头

    # 构造方法  使用构造方法可以不用定义上面的成员变量
    def __init__(self, name, gender, nationality, native, age):
        self.name = name
        self.gender = gender
        self.nationality = nationality
        self.native = native
        self.age = 411

    # 重写魔术方法
    #tostr
    def __str__(self, name, gender, nationality, native, age):
        return f"name:{self.name},gender:{gender},nationality:{nationality},native:{self.native},age:{self.age}"
    #小于   若是小于等于 则用 __le__
    def __lt__(self, other):
        return self.age < other.age
    # 等于
    def __eq__(self, other):
        return self.age == other.age

    # self 代表类对象本身
    def print_info(self, msg):
        print(f"大家好,我是{self.name},我来自{self.native},我今年{self.age}岁,{msg}")

    #私有成员方法
    def __private_method(self):
        print("我是私有成员方法")

stu1 = Student("James", "男", "汉", "河南省郑州市", 18)
stu1.print_info("很高兴认识大家")
