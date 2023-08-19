# Author : 佟星TONGXING
# Date : 2022/11/16 11:16
# Version: 1.0

from typing import Union
# 定义联合类型注解
my_dict : dict[str,Union[str,int]] = {"name":"Tom","age":18}
def func_union(data:Union[int,str])-> Union[int,str]:
    pass

class Phone:
    IMEI = None
    producer = "富士康"
    def call_by_4g(self):
        print("father:4g.......")

class NFC:
    nfc_type = "第五代"
    nfc_producer = "AMG"

# 多继承
class iPhone_13(Phone,NFC):
    face_id = True
    producer = "意尔康"

    # super().xx   调用父类的成员变量和成员方法
    def print_father(self):
        print(f"1父类的厂商producer:{NFC.nfc_producer}")
        print(f"2父类的厂商producer:{super().producer}")
        super().call_by_4g()
    # 重写父类方法
    def call_by_4g(self):
        print("child：4g.........")
    def call_by_5g(self):
        print("5g.......")



phone_13 = iPhone_13()
phone_13.print_father()
