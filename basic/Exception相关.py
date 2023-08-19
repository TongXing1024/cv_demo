"""
异常具有传递性
"""
def catch_exception():
    try:
        i = 1/0
    # except Exception as e      捕获所有异常
    except ZeroDivisionError as e:   #捕获指定异常
        print(e)
    else:
        print("我是else,这是没有异常时执行的代码")
    finally:
        print("我是finally,无论如何,我都会执行")   #可以用来关闭文件

if __name__ == '__main__':
    catch_exception()