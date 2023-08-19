# Author : 佟星TONGXING
# Date : 2022/11/13 19:55
# Version: 1.0

if int(input("请输入您的身高:")) > 120:
    print("很抱歉,您的身高超过了最低限制,要买成人票")
    print("但是如果你的vip等级超过3级,也可以免费玩")
    if int(input("请输入您的vip等级:")) >3:
        print("恭喜你,你的等级可以免费玩")
    else:
        print("很抱歉,你的等级不可以免费玩")
else:
    print("恭喜你,可以免费玩耍!")
