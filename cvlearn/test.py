import cv2 as cv
import numpy as np

arr = np.array([
    [1,2,3],
    [4,5,6,],
    [7,8,9]
])
print(arr.shape)

rows,clos = arr.shape
row_num = 0
clo_num = 0
total = 0
for i in range(rows):
    for j in range(clos):
        total = total + arr[i][j]
        clo_num = clo_num + (1+j) * arr[i][j]
        row_num = row_num + (i+1) * arr[i][j]

print(f"total:{total}")
print(f"clo_num:{clo_num}")
print(f"row_num:{row_num}")

# it = 10
# for i in range(it):
#     print(i)