import numpy as np


my_list = ['google','baidu','edge','firefox','chatGPT','baidu']
li = [[['google','baidu','a'],['edge','firefox','b'],['chatGPT','baidu','c']]]
num = [[[1,2,3],[4,5,6],[7,8,9]]]


data = np.array(num)


print(f"data.shape:{data.shape}")
a = data[:,-1,:]
print(f"data[:,0,0]:{data[:,0,0]}") #1
print(f"data[:,0,:]:{data[:,0,:]}") #1 2 3
print(f"data[:,1,:]:{data[:,1,:]}") # 4 5 6
print(f"data[:,-1,:]:{data[:,-1,:]}") # 7 8 9

print(f"data[:,:,2]:{data[:,:,0:2]},dtype:{data[:,:,0:2].shape}")



# print(my_list[0 : -2])
# print(len(my_list))
# print(my_list.count('baidu'))


my_tup = (1,2,3,4,5,6,7)
for x in my_tup:
    print(x,end= " ")
