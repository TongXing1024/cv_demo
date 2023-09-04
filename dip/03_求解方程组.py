# @Time    : 2023/9/1 11:31
# @Author  : TONGXING
import math
import sympy as sp
import numpy as np
from scipy.optimize import fsolve
from sympy.core.evalf import evalf


# 定义一个包含两个非线性方程的函数
def equations(x, k, b1, a, b2, c):
    # 定义两个方程
    eq1 = k * x[0] + b1 - x[1]
    eq2 = a * x[0] ** 2 + b2 * x[0] + c - x[1]
    return [eq1, eq2]
def get_intersection_point2(straight_lineParams,boundary_lineParams):
    """
    此函数用于获取两条直线的交点
    params1:直线参数1
    params2:曲线参数2
    """
    # 直线参数 y = kx + b
    k = straight_lineParams[0]
    b1 = straight_lineParams[1]
    # 曲线参数  y = ax^2 + bx + c
    a,b2,c = boundary_lineParams[0],boundary_lineParams[1],boundary_lineParams[2]
    # 初始猜测值
    initial_guess = [1.0, 2.0]
    # 使用 fsolve 求解方程组
    solution = fsolve(equations, initial_guess, args=(k, b1, a, b2, c))
    # 取绝对值
    x = solution[0]
    y = k * x + b1
    intersection_point = [x, y]
    return intersection_point
def get_intersection_point(straight_lineParams,boundary_lineParams):
    """
    此函数用于获取两条直线的交点
    params1:直线参数1
    params2:曲线参数2
    """
    # 直线参数 y = kx + b
    k = straight_lineParams[0]
    b1 = straight_lineParams[1]
    # print(f"直线方程为：y={k}x+{b1}")
    # 曲线参数  y = ax^2 + bx + c
    a,b2,c = boundary_lineParams[0],boundary_lineParams[1],boundary_lineParams[2]
    # print(f"曲线方程为：y={a}x^2+{b2}x+{c}")
    # 定义变量
    x = sp.symbols('x')
    # 合并同类项
    eq_linear = a*x**2 + (b2-k)*x + (c-b1)
    print("合并同类项后的方程为：",eq_linear)
    solution_linear = sp.solve(eq_linear, x)
    # 找到大于0的解
    for i in range(len(solution_linear)):
        if solution_linear[i] > 0:
            solution_linear = solution_linear[i]
            break
    x = solution_linear
    y = k * x + b1
    intersection_point = np.array([x,y])
    return intersection_point


# 验证是否满足方程
def test_f(straight_lineParams,boundary_lineParams,intersection_point):
    k, b1 = straight_lineParams[0], straight_lineParams[1]
    print(f"直线方程为：y={k}x+{b1}")
    a, b2, c = boundary_lineParams[0], boundary_lineParams[1], boundary_lineParams[2]
    print(f"曲线方程为：y={a}x^2+{b2}x+{c}")
    # 拿到两个解
    x1 = intersection_point[0]
    x2 = intersection_point[1]
    # 根据x1构造方程
    eq1 = k * x1 + b1
    eq2 = a * x1 ** 2 + b2 * x1 + c
    # 根据x2构造方程
    eq3 = k * x2 + b1
    eq4 = a * x2 ** 2 + b2 * x2 + c
    if math.isclose(eq1, eq2, rel_tol=1e-5):
        if math.isclose(eq3, eq4, rel_tol=1e-5):
            print("方程验证成功！，交点为：",(x1,eq1),(x2,eq3))
    else:
        print("方程验证失败！")

if __name__ == '__main__':
    straight_lineParams = np.array([0.07547237, 526.1389])
    boundary_lineParams = np.array([0.0002213496496011766, -0.02665122730875861, 411.688625376784])
    # straight_lineParams = np.array([1, 0])
    # boundary_lineParams = np.array([1, 1, -12])
    intersection_point = get_intersection_point(straight_lineParams, boundary_lineParams)
    print("交点为：",intersection_point)
    # test_f(straight_lineParams, boundary_lineParams,intersection_point)