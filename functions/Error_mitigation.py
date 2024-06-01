import pennylane as qml
from pennylane import numpy as np



def extra_polation(circ,H,theta,noise_factor=[0,1,2,3],p1=0.01,p2=0.02):
    
    real_vaule = circ(theta)
    dev = circ.device
    ops = circ.qtape.operations
    ops_inv = ops[::-1]


    
    def noise_circ():
        tensor = torch.tensor
        for op in ops:
            eval(f'qml.{op}')
            if len(op.wires)>1:
                for wire in op.wires:
                    qml.DepolarizingChannel(p1, wires=wire)
                    qml.BitFlip(p2, wires=wire)
            else:
                qml.DepolarizingChannel(p1, wires=op.wires) 
                qml.BitFlip(p2, wires=op.wires)
    def noise_circ_inv():
        tensor = torch.tensor
        for op in ops_inv:
            eval(f'qml.adjoint(qml.{op})')
            if len(op.wires)>1:
                for wire in op.wires:
                    qml.DepolarizingChannel(p1, wires=wire)
                    qml.BitFlip(p2, wires=wire)
            else:
                qml.DepolarizingChannel(p1, wires=op.wires) 
                qml.BitFlip(p2, wires=op.wires)
    
    @qml.qnode(dev,interface='torch')
    def extra_polation(factor):
        PauliZ = qml.PauliZ
        PauliY = qml.PauliY
        PauliX = qml.PauliX
        for i in range(factor):
            noise_circ()
            noise_circ_inv()
        noise_circ()
        return qml.expval(H)
    




    res = []
    for factor in noise_factor:
        res.append(extra_polation(factor).detach().numpy())
    return res,real_vaule
    






import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 선형 함수 정의
def linear_func(x, a, b):
    return a * x + b

# 지수 함수 정의
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# 데이터 피팅 및 외삽 함수
def extrapolate_values(y_data,x_data = np.array([1, 3, 5, 7])):
    # 선형 함수 피팅
    params_linear, _ = curve_fit(linear_func, x_data, y_data)
    a_linear, b_linear = params_linear
    y_linear_0 = linear_func(0, a_linear, b_linear)

    # 지수 함수 피팅 초기 추정값 및 범위 설정
    initial_guess = [1, 1, 1]
    bounds = ([0, -1, -np.inf], [np.inf, 1, np.inf])

    params_exp, _ = curve_fit(exp_func, x_data, y_data, p0=initial_guess, bounds=bounds, maxfev=10000)
    a_exp, b_exp, c_exp = params_exp
    y_exp_0 = exp_func(0, a_exp, b_exp, c_exp)

    return y_linear_0, y_exp_0, params_linear, params_exp
