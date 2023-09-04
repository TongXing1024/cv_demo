# @Time    : 2023/8/21 16:43
# @Author  : TONGXING

import sympy as sp

# 定义变量
x = sp.symbols('x')
# 一元一次方程：y = 2x + 3
eq_linear = (x - 3) * (x - 4)
solution_linear = sp.solve(eq_linear, x)
print(eq_linear)
print(solution_linear)

# # 一元二次方程：y = x^2 - 4x + 4
# eq_quadratic = sp.Eq(x**2 - 4*x + 4, x)
#
# # 解方程

# solution_quadratic = sp.solve(eq_quadratic, x)
#
# print(solution_linear)
# print(solution_quadratic)