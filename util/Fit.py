# @Time    : 2023/8/11 16:37
# @Author  : TONGXING
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

class FitC:
    """
    ransac算法拟合直线
    """
    def fit_line_ransac(self,points, num_iterations, threshold):
        """
        :param points: 点的坐标 shape:(n,2)，n为点的个数，每个点的坐标为(x,y)
        :param num_iterations:  迭代次数
        :param threshold:   阈值 通常为0.1
        :return: best_line: 最佳拟合直线的参数 ； best_inliers: 内点
        """
        best_line = None  # 最佳拟合直线的参数
        best_inliers = []  # 内点
        for _ in range(num_iterations):  # 迭代次数
            # 随机选择两个点形成候选直线
            sample_indices = np.random.choice(len(points), 2, replace=False)  # sample_indices为随机选择的两个点的索引
            p1, p2 = points[sample_indices]
            # 使用选择的两个点拟合一条直线
            line = np.polyfit([p1[0], p2[0]], [p1[1], p2[1]],
                              1)  # np.polyfit(x,y,1)返回的是一次函数的系数，即斜率和截距,这里的x和y是两个点的坐标,1表示一次函数
            # 计算所有点到直线的距离
            distances = np.abs(np.polyval(line, points[:, 0]) - points[:, 1])
            # 统计内点（距离在阈值内的点）
            inliers = points[distances < threshold]

            if len(inliers) > len(best_inliers):
                best_line = line
                best_inliers = inliers

        return best_line, best_inliers

def test_ransac():
    """
    测试RANSAC算法
    :return:
    """
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 5 + np.random.normal(0, 2, 100)  # y = 2x + 5
    # 添加一些异常值
    outlier_indices = np.random.choice(len(x), 20, replace=False)
    y[outlier_indices] += np.random.normal(0, 50, 20)
    # 将X和Y合并成一个100x2的矩阵，每一行表示一个数据点的X和Y坐标
    points = np.column_stack((x, y))
    fit = Fit()
    best_line, best_inliers = fit.fit_line_ransac(points, 100, 0.1)
    print(best_line)