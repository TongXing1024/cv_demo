# @Time    : 2023/8/3 9:57
# @Author  : TONGXING
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def fit_line_ransac(points, num_iterations=100, threshold=0.1):
    best_line = None # 最佳拟合直线的参数
    best_inliers = [] # 内点
    for _ in range(num_iterations): # 迭代次数
        # 随机选择两个点形成候选直线
        sample_indices = np.random.choice(len(points), 2, replace=False) #sample_indices为随机选择的两个点的索引
        p1, p2 = points[sample_indices]
        print(len(p1))
        # 使用选择的两个点拟合一条直线
        line = np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1)    #np.polyfit(x,y,1)返回的是一次函数的系数，即斜率和截距,这里的x和y是两个点的坐标,1表示一次函数

        # 计算所有点到直线的距离
        distances = np.abs(np.polyval(line, points[:, 0]) - points[:, 1])
        print(f"dis:{distances}")
        # 统计内点（距离在阈值内的点）
        inliers = points[distances < threshold]

        if len(inliers) > len(best_inliers):
            best_line = line
            best_inliers = inliers

    return best_line, best_inliers


# 示例用法：
# 生成一些围绕一条直线的带噪声的随机数据点
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2 * x + 5 + np.random.normal(0, 2, 100) # y = 2x + 5

# 添加一些异常值
outlier_indices = np.random.choice(len(x), 20, replace=False)
y[outlier_indices] += np.random.normal(0, 50, 20)

# 将X和Y合并成一个100x2的矩阵，每一行表示一个数据点的X和Y坐标
points = np.column_stack((x, y))


# 使用RANSAC拟合直线
best_line, best_inliers = fit_line_ransac(points)

# 绘制原始数据点
plt.scatter(points[:, 0], points[:, 1], label='datapoint') #

# 绘制用于拟合直线的内点
plt.scatter(best_inliers[:, 0], best_inliers[:, 1], color='red', label='innerPoit')

# 绘制拟合的直线
plt.plot(x, np.polyval(best_line, x), color='orange', linewidth=3, label='line')

# 设置图表的标签和网格
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()

# 显示图表
plt.show()


